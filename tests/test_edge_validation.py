from __future__ import annotations

import copy
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.edge_validation import (
    EdgeRegionMetric, EdgeValidationProtocol, masked_rgb_psnr_per_image, select_epochs, weighted_score,
)
from src.trainer import Trainer


def protocol_config(**overrides):
    return {"training": {"loss": "l1"}, "validation": {"edge_evaluation": {"enabled": True, **overrides}}}


def history_row(epoch, psnr, edge, alpha=0.3):
    return {"epoch": epoch, "val_psnr_db": psnr, "val_edge_region_psnr_db": edge,
            "val_weighted_score": weighted_score(psnr, edge, alpha)}


class EdgeMetricTests(unittest.TestCase):
    def test_known_rgb_error_is_normalized_by_mask_pixels(self):
        target = torch.zeros(1, 3, 12, 14)
        mask = torch.zeros(1, 1, 12, 14, dtype=torch.bool)
        mask[:, :, 3:7, 4:8] = True
        prediction = torch.ones_like(target)
        prediction[:, :, 3:7, 4:8] = 0.1
        result = masked_rgb_psnr_per_image(prediction, target, mask)
        self.assertAlmostEqual(result["psnr_db"].item(), 20.0, places=5)
        self.assertEqual(result["pixels"].item(), 16)
        prediction[:, 1:, 3:7, 4:8] = 0
        self.assertAlmostEqual(masked_rgb_psnr_per_image(prediction, target, mask)["psnr_db"].item(),
                               20 + 10 * math.log10(3), places=5)

    def test_known_step_sobel_and_dilation_exclude_outer_border(self):
        target = torch.zeros(1, 3, 16, 16)
        target[:, :, :, 8:] = 1
        for radius, expected_columns in ((0, 2), (1, 4)):
            metric = EdgeRegionMetric(EdgeValidationProtocol(enabled=True, sobel_threshold=0.2, dilation_radius=radius))
            mask = metric.mask(target)
            self.assertEqual(mask.sum().item(), 14 * expected_columns)
            self.assertFalse(mask[:, :, (0, -1), :].any())
            self.assertFalse(mask[:, :, :, (0, -1)].any())

    def test_empty_mask_is_missing_not_perfect(self):
        image = torch.zeros(1, 3, 16, 16)
        result = EdgeRegionMetric(EdgeValidationProtocol(enabled=True))(image, image)
        self.assertEqual(result["pixels"].item(), 0)
        self.assertTrue(torch.isnan(result["psnr_db"]).all())

    def test_mask_depends_only_on_hr_and_perfect_nonempty_is_infinite(self):
        reference = torch.zeros(1, 3, 16, 16)
        reference[:, :, :, 8:] = 1
        metric = EdgeRegionMetric(EdgeValidationProtocol(enabled=True))
        perfect = metric(reference, reference)
        blurred = metric(torch.full_like(reference, 0.5), reference)
        self.assertEqual(perfect["pixels"].item(), blurred["pixels"].item())
        self.assertTrue(torch.isposinf(perfect["psnr_db"]).all())
        self.assertAlmostEqual(blurred["psnr_db"].item(), 6.0206, places=4)

    def test_invalid_protocols_are_rejected(self):
        for values in ({"weighted_alpha": 1.1}, {"psnr_tolerance_db": -0.1},
                       {"sobel_threshold": 0}, {"dilation_radius": 1.5},
                       {"enabled": "true"}, {"weighted_alhpa": 0.3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EdgeValidationProtocol.from_config(protocol_config(**values))


class SelectionTests(unittest.TestCase):
    def test_weighted_choice_differs_from_psnr(self):
        choices = select_epochs([history_row(1, 30, 24), history_row(2, 29.9, 24.5)], EdgeValidationProtocol())
        self.assertEqual(choices["psnr"]["epoch"], 1)
        self.assertEqual(choices["weighted"]["epoch"], 2)
        self.assertAlmostEqual(choices["weighted"]["val_weighted_score"], 28.28)

    def test_constrained_choice_can_return_to_an_earlier_candidate(self):
        history = [history_row(1, 30, 24), history_row(2, 29.95, 25)]
        self.assertEqual(select_epochs(history, EdgeValidationProtocol())["edge_constrained"]["epoch"], 2)
        history.append(history_row(3, 30.08, 23))
        self.assertEqual(select_epochs(history, EdgeValidationProtocol())["edge_constrained"]["epoch"], 1)

    def test_ties_are_stable_and_alpha_endpoints_handle_infinity(self):
        history = [history_row(1, 30, 24), history_row(2, 30, 24)]
        self.assertTrue(all(row["epoch"] == 1 for row in select_epochs(history, EdgeValidationProtocol()).values()))
        self.assertEqual(weighted_score(30, float("inf"), 0), 30)
        self.assertEqual(weighted_score(float("inf"), 24, 1), 24)


class TinySR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2))

    def forward(self, image):
        return self.model(image)


class TrainingIntegrationTests(unittest.TestCase):
    def trainer(self, root, model, config, epochs=2):
        logger = logging.getLogger("edge-tests")
        logger.handlers = [logging.NullHandler()]
        return Trainer(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                       device=torch.device("cpu"), epochs=epochs, scale_factor=2, config=config,
                       run_name="test", pretrained_checkpoint=root / "unused.pth",
                       checkpoint_dir=root / "checkpoints", metric_dir=root / "metrics",
                       sample_dir=root / "samples", logger=logger, validation_is_independent=True,
                       save_samples=False)

    def loader(self):
        generator = torch.Generator().manual_seed(93)
        return DataLoader([{"lr": torch.rand(3, 8, 8, generator=generator),
                            "hr": torch.rand(3, 16, 16, generator=generator), "name": f"frame{i}"}
                           for i in range(2)], batch_size=1)

    def test_extra_validation_leaves_l1_training_updates_identical(self):
        torch.manual_seed(12)
        initial = TinySR()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.trainer(root / "baseline", copy.deepcopy(initial), {})
            enhanced = self.trainer(root / "enhanced", copy.deepcopy(initial), protocol_config())
            baseline.fit(self.loader(), self.loader())
            enhanced.fit(self.loader(), self.loader())
            for a, b in zip(baseline.model.parameters(), enhanced.model.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual([r["train_loss"] for r in baseline.history], [r["train_loss"] for r in enhanced.history])
            self.assertEqual([r["val_psnr_db"] for r in baseline.history], [r["val_psnr_db"] for r in enhanced.history])
            summary = json.loads((enhanced.metric_dir / "selection_summary.json").read_text())
            self.assertEqual(len(summary["selected"]), 3)
            for choice in summary["selected"].values():
                checkpoint = torch.load(choice["checkpoint"], weights_only=True)
                self.assertEqual(checkpoint["epoch"], choice["epoch"])

    def test_deferred_ssim_keeps_training_scores_and_selection_identical(self):
        torch.manual_seed(12)
        initial = TinySR()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_epoch = self.trainer(root / "per_epoch", copy.deepcopy(initial), protocol_config())
            config = protocol_config()
            config["validation"]["ssim_schedule"] = "post_selection"
            deferred = self.trainer(root / "deferred", copy.deepcopy(initial), config)
            per_epoch.fit(self.loader(), self.loader())
            with patch("src.trainer.rgb_ssim", side_effect=AssertionError("SSIM must not run inside an epoch")):
                deferred.fit(self.loader(), self.loader())
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(per_epoch.model.parameters(), deferred.model.parameters())))
            for original, changed in zip(per_epoch.history, deferred.history):
                for key in ("train_loss", "val_psnr_db", "val_edge_region_psnr_db", "val_weighted_score"):
                    self.assertEqual(original[key], changed[key])
                self.assertIsNone(changed["val_ssim"])
                self.assertEqual(changed["val_ssim_status"], "deferred_post_selection")
            a = select_epochs(per_epoch.history, per_epoch.edge_protocol)
            b = select_epochs(deferred.history, deferred.edge_protocol)
            self.assertEqual({k: v["epoch"] for k, v in a.items()}, {k: v["epoch"] for k, v in b.items()})

    def test_resume_allows_ssim_reporting_revision_and_preserves_old_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.trainer(root, TinySR(), protocol_config(), epochs=1)
            first.fit(self.loader(), self.loader())
            config = protocol_config()
            config["validation"]["ssim_schedule"] = "post_selection"
            resumed = self.trainer(root, TinySR(), config, epochs=2)
            saved_ssim = first.history[0]["val_ssim"]
            resumed.fit(self.loader(), self.loader(), start_epoch=resumed.resume(first.checkpoint_dir / "last.pth"))
            self.assertEqual(resumed.history[0]["val_ssim"], saved_ssim)
            self.assertEqual(resumed.history[0]["val_ssim_status"], "per_epoch")
            self.assertIsNone(resumed.history[1]["val_ssim"])
            self.assertEqual(resumed.history[1]["resumed_from_epoch"], 1)

    def test_resume_restores_comparison_and_refuses_protocol_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.trainer(root, TinySR(), protocol_config(), epochs=1)
            original.fit(self.loader(), self.loader())
            path = original.checkpoint_dir / "last.pth"
            invalid = self.trainer(root, TinySR(), protocol_config(weighted_alpha=0.5))
            with self.assertRaisesRegex(ValueError, "protocol"):
                invalid.resume(path)
            changed_data = self.trainer(root, TinySR(), protocol_config())
            changed_data.validation_fingerprint = "different-validation-images"
            with self.assertRaisesRegex(ValueError, "contents changed"):
                changed_data.resume(path)
            resumed = self.trainer(root, TinySR(), protocol_config(), epochs=2)
            epoch = resumed.resume(path)
            resumed.fit(self.loader(), self.loader(), start_epoch=epoch)
            self.assertEqual([r["epoch"] for r in resumed.history], [1, 2])
            self.assertEqual(resumed.history[-1]["resumed_from_epoch"], 1)
            self.assertEqual(json.loads((resumed.metric_dir / "selection_summary.json").read_text())["completed_epoch"], 2)

    def test_constrained_checkpoint_recovers_a_previous_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self.trainer(Path(temporary), TinySR(), protocol_config(), epochs=3)
            for row in [history_row(1, 30, 24), history_row(2, 29.95, 25), history_row(3, 30.08, 23)]:
                trainer.history.append(row)
                trainer._save_selection_comparison(row["epoch"])
            chosen = torch.load(trainer.checkpoint_dir / "best_edge_constrained.pth", weights_only=True)
            self.assertEqual(chosen["epoch"], 1)

    def test_mixed_batch_excludes_only_empty_masks_from_edge_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self.trainer(Path(temporary), TinySR(), protocol_config())
            items = list(self.loader().dataset)
            items[0]["hr"] = torch.zeros_like(items[0]["hr"])
            loader = DataLoader(items, batch_size=2)
            trainer.validate(loader, 1)
            self.assertEqual(trainer.last_validation_details["val_edge_images"], 1)
            with torch.inference_mode():
                output = trainer.model(items[1]["lr"].unsqueeze(0))
                expected = trainer.edge_metric(output, items[1]["hr"].unsqueeze(0))["psnr_db"].item()
            self.assertAlmostEqual(trainer.last_validation_details["val_edge_region_psnr_db"], expected, places=5)

    def test_legacy_l1_checkpoint_without_protocol_still_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.trainer(root, TinySR(), {}, epochs=1)
            original.fit(self.loader(), self.loader())
            path = original.checkpoint_dir / "last.pth"
            checkpoint = torch.load(path, weights_only=True)
            checkpoint.pop("edge_validation_protocol")
            torch.save(checkpoint, path)
            resumed = self.trainer(root, TinySR(), {}, epochs=2)
            self.assertEqual(resumed.resume(path), 1)


class OutputProtectionTests(unittest.TestCase):
    def test_new_runs_cannot_reuse_an_existing_output_directory(self):
        spec = importlib.util.spec_from_file_location("finetune_entry", ROOT / "scripts/03_finetune.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                entry._check_output_directories([existing], None)


if __name__ == "__main__":
    unittest.main()
