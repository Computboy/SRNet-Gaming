from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edge_validation import EdgeValidationProtocol, EdgeRegionMetric
from src.metrics import rgb_psnr_per_image, rgb_ssim, ssim_schedule_from_config
from src.post_selection import evaluate_rows, merge_reported_ssim, selected_ssim


class TinySR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2))

    def forward(self, image):
        return self.model(image)


class PostSelectionTests(unittest.TestCase):
    def test_reporting_schedule_is_validated(self):
        self.assertEqual(ssim_schedule_from_config({}), "per_epoch")
        for value in ("skip", False, [], None):
            with self.assertRaises(ValueError):
                ssim_schedule_from_config({"validation": {"ssim_schedule": value}})

    def test_posthoc_matches_original_full_image_metrics_without_weight_updates(self):
        generator = torch.Generator().manual_seed(41)
        items = [{"lr": torch.rand(3, 8, 8, generator=generator),
                  "hr": torch.rand(3, 16, 16, generator=generator), "name": f"frame{i}"} for i in range(2)]
        items[0]["hr"].zero_()
        loader = DataLoader(items, batch_size=1)
        model = TinySR().eval()
        before = copy.deepcopy(model.state_dict())
        protocol = EdgeValidationProtocol(enabled=True)
        original = []
        with torch.inference_mode():
            for batch in loader:
                sr, hr = model(batch["lr"]), batch["hr"]
                edge = EdgeRegionMetric(protocol)(sr, hr)
                original.append({"key": batch["name"][0], "psnr_db": rgb_psnr_per_image(sr, hr)[0].item(),
                    "ssim": rgb_ssim(sr[0].clamp(0, 1).permute(1, 2, 0).numpy(), hr[0].permute(1, 2, 0).numpy()),
                    "edge_region_psnr_db": edge["psnr_db"][0].item() if edge["pixels"][0].item() else None})
        evaluated = evaluate_rows(model, loader, torch.device("cpu"), protocol, {r["key"] for r in original})
        self.assertEqual(original, evaluated)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        deferred = [{**row, "ssim": ""} for row in original]
        self.assertEqual(merge_reported_ssim(deferred, evaluated), original)
        changed = copy.deepcopy(evaluated)
        changed[0]["psnr_db"] += 1
        with self.assertRaisesRegex(ValueError, "differs"):
            merge_reported_ssim(deferred, changed)
        with self.assertRaisesRegex(ValueError, "keys"):
            merge_reported_ssim(deferred, evaluated[:-1])
        with self.assertRaisesRegex(ValueError, "complete"):
            evaluate_rows(model, loader, torch.device("cpu"), protocol, {"wrong"})

    def test_three_selectors_share_one_posthoc_evaluation_for_same_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrained = root / "official.pth"
            pretrained.write_bytes(b"fixture")
            model = TinySR()
            checkpoint_path = root / "selected.pth"
            torch.save({"epoch": 1, "state_dict": model.model.state_dict(), "validation_fingerprint": "fixed"}, checkpoint_path)
            rows = [{"key": f"scene_1_02/frame_{i:04d}", "psnr_db": 20., "ssim": .8, "edge_region_psnr_db": 18.}
                    for i in range(240)]
            original = [{**r, "ssim": ""} for r in rows]
            config = {"data": {"val_lr": "lr", "val_hr": "hr"},
                      "model": {"pretrained_checkpoint": "official.pth"},
                      "validation": {"edge_evaluation": {"enabled": True}},
                      "runtime": {"pretrained_checkpoint_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest()}}
            data = {"run": "example", "checkpoint": {"config": config},
                    "manifest": {"sha256": "fixed", "pairs": [{"key": r["key"]} for r in rows]},
                    "summary": {"selected": {name: {"epoch": 1, "checkpoint": str(checkpoint_path)}
                                              for name in ("psnr", "weighted", "edge_constrained")}}}
            def save_csv(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            with patch("src.post_selection.PairedImageDataset", return_value=[]), \
                 patch("src.post_selection.verify_manifest"), \
                 patch("src.post_selection.evaluate_rows", return_value=rows) as evaluate:
                result = selected_ssim(data, root, root / "report", lambda _: original, save_csv, lambda _: TinySR())
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(list(result), [1])
            metadata = json.loads((root / "report/epoch001.json").read_text())
            self.assertEqual(metadata["frames"], 240)
            self.assertAlmostEqual(metadata["mean_ssim"], .8)
            self.assertFalse(metadata["test_accessed"])


if __name__ == "__main__":
    unittest.main()
