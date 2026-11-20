from __future__ import annotations

import copy
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml_config


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ABProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = module("ab_runner", "10_run_edge_l1_ab.py")
        cls.analysis = module("ab_analysis", "11_compare_edge_l1_ab.py")
        cls.a = load_yaml_config(ROOT / "configs/asian_village_selection.yaml")
        cls.b = load_yaml_config(ROOT / "configs/asian_village_edge_l1.yaml")

    def test_declared_configs_differ_only_in_training_loss(self):
        self.runner.verify_pair(self.a, self.b)
        self.analysis.verify_comparable_configs(self.a, self.b)

    def test_other_training_or_validation_changes_are_rejected(self):
        for section, key, value in (("training", "learning_rate", 1e-3), ("training", "batch_size", 8),
                                    ("training", "edge_l1_weight", 0.2), ("data", "lr_patch_size", 64),
                                    ("model", "pretrained_checkpoint", "different.pth")):
            changed = copy.deepcopy(self.b)
            changed[section][key] = value
            for check in (self.runner.verify_pair, self.analysis.verify_comparable_configs):
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    check(self.a, changed)
        changed = copy.deepcopy(self.b)
        changed["validation"]["edge_evaluation"]["weighted_alpha"] = 0.5
        for check in (self.runner.verify_pair, self.analysis.verify_comparable_configs):
            with self.assertRaises(ValueError):
                check(self.a, changed)

    def test_swapping_a_and_b_is_not_a_valid_baseline_comparison(self):
        for check in (self.runner.verify_pair, self.analysis.verify_comparable_configs):
            with self.assertRaises(ValueError):
                check(self.b, self.a)

    def test_revision_only_allows_ssim_schedule_change(self):
        old = copy.deepcopy(self.a)
        old["validation"].pop("ssim_schedule")
        self.runner.verify_ssim_revision(old, self.a)
        changed = copy.deepcopy(self.a)
        changed["training"]["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            self.runner.verify_ssim_revision(old, changed)

    def test_selected_validation_requires_all_240_frames_and_matching_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = self.analysis.ROOT
            self.analysis.ROOT = root
            try:
                directory = root / "outputs/metrics/example/validation"
                directory.mkdir(parents=True)
                path = directory / "epoch001.csv"
                rows = [{"key": f"scene_1_02/frame_{i:04d}", "psnr_db": 20., "ssim": 0.8,
                         "edge_region_psnr_db": 18.} for i in range(240)]
                def write(items):
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(items)
                write(rows)
                weight = root / "selected.pth"
                torch.save({"epoch": 1}, weight)
                choice = {"epoch": 1, "val_psnr_db": 20., "val_ssim": 0.8,
                          "val_edge_region_psnr_db": 18., "val_weighted_score": 19.4, "checkpoint": str(weight)}
                data = {"run": "example", "manifest": {"pairs": [{"key": row["key"]} for row in rows]},
                        "summary": {"selected": {name: dict(choice) for name in ("psnr", "weighted", "edge_constrained")}}}
                records, mapping = self.analysis.selected_validation(data, "A")
                self.assertEqual(len(records), 3)
                self.assertAlmostEqual(records[0]["val_psnr_db"], 20.)
                self.assertEqual(len(mapping["psnr"]), 240)
                deferred = [{**row, "ssim": ""} for row in rows]
                write(deferred)
                for choice in data["summary"]["selected"].values():
                    choice["val_ssim"] = None
                with self.assertRaisesRegex(ValueError, "SSIM"):
                    self.analysis.selected_validation(data, "A")
                data["post_selection_ssim"] = {1: {"rows": rows}}
                records, _ = self.analysis.selected_validation(data, "A")
                self.assertAlmostEqual(records[0]["val_ssim"], .8)
                self.assertEqual(records[0]["ssim_source"], "post_selection")
                write(rows[:-1])
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    self.analysis.selected_validation(data, "A")
            finally:
                self.analysis.ROOT = old_root

    def test_ab_plot_renderer_handles_training_loss_components(self):
        history = [{"epoch": i, "val_psnr_db": 25 + i, "val_edge_region_psnr_db": 20 + i,
                    "val_ssim": 0.8 + i / 100, "val_weighted_score": 23 + i,
                    "train_global_l1": 0.02 / i, "train_edge_l1": 0.04 / i, "train_total_loss": 0.024 / i}
                   for i in range(1, 4)]
        data = {"history": history, "summary": {"selected": {name: {"epoch": 3}
                for name in ("psnr", "weighted", "edge_constrained")}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.analysis.plot_curves(data, data, root)
            for name in ("validation_comparison.png", "validation_comparison.svg", "loss_decomposition.png", "loss_decomposition.svg"):
                self.assertGreater((root / name).stat().st_size, 1000)
            records = [{"experiment": label, "strategy": strategy, "val_ssim": .8}
                       for label in ("A", "B") for strategy in ("psnr", "weighted", "edge_constrained")]
            self.analysis.plot_selected_ssim(records, root)
            self.assertGreater((root / "selected_ssim_comparison.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
