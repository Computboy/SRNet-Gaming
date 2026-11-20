from __future__ import annotations
import copy
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from src.capacity import benchmark_pair, model_description, verify_capacity_configs
from src.trainer import (Trainer, build_optimizer, build_quicksrnet_from_local_checkpoint,
                         build_quicksrnet_small_from_local_checkpoint, verify_checkpoint_reload)
from src.utils import load_yaml_config


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = load_yaml_config(ROOT / "configs/asian_village_selection.yaml")
        cls.c = load_yaml_config(ROOT / "configs/asian_village_medium_l1.yaml")
        cls.small = ROOT / cls.a["model"]["pretrained_checkpoint"]
        cls.medium = ROOT / cls.c["model"]["pretrained_checkpoint"]

    def test_configs_change_only_capacity_initialization_and_output_identity(self):
        result = verify_capacity_configs(self.a, self.c)
        self.assertEqual(len(result["changed"]), 3)
        self.assertEqual(self.c["training"]["edge_l1_weight"], 0)

    def test_non_capacity_training_and_selection_changes_are_rejected(self):
        for section, field, value in (("training", "loss", "l1_edge"), ("training", "edge_l1_weight", .1),
                ("training", "batch_size", 8), ("training", "learning_rate", 2e-4), ("training", "epochs", 50),
                ("training", "amp", False), ("data", "lr_patch_size", 64), ("validation", "ssim_schedule", "per_epoch"),
                ("model", "pretrained_sha256", "wrong")):
            changed = copy.deepcopy(self.c)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                verify_capacity_configs(self.a, changed)
        for field, value in (("sobel_threshold", .1), ("dilation_radius", 2), ("weighted_alpha", .5), ("psnr_tolerance_db", .2)):
            changed = copy.deepcopy(self.c)
            changed["validation"]["edge_evaluation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_capacity_configs(self.a, changed)

    def test_official_medium_loads_every_tensor_and_has_expected_structure(self):
        model = build_quicksrnet_from_local_checkpoint(self.medium, model_name="quicksrnet_medium")
        official = torch.load(self.medium, map_location="cpu", weights_only=False)["state_dict"]
        self.assertEqual(set(model.model.state_dict()), set(official))
        self.assertTrue(all(torch.equal(value, official[key]) for key, value in model.model.state_dict().items()))
        info = model_description(model)
        self.assertEqual(info["parameters"], 50604)
        self.assertEqual(info["conv_layers"], 7)
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.rand(2, 3, 16, 17)).shape), (2, 3, 32, 34))

    def test_small_api_and_tensor_values_are_unchanged(self):
        old_api = build_quicksrnet_small_from_local_checkpoint(self.small)
        generic = build_quicksrnet_from_local_checkpoint(self.small)
        self.assertEqual(model_description(old_api)["parameters"], 22860)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_api.parameters(), generic.parameters())))
        sample = torch.rand(1, 3, 16, 17)
        with torch.inference_mode():
            self.assertTrue(torch.equal(old_api(sample), generic(sample)))

    def test_small_weights_and_unknown_variants_are_rejected_for_medium(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            build_quicksrnet_from_local_checkpoint(self.small, model_name="quicksrnet_medium")
        with self.assertRaises(ValueError):
            build_quicksrnet_from_local_checkpoint(self.medium, model_name="unknown")
        with self.assertRaises(ValueError):
            build_quicksrnet_from_local_checkpoint(self.medium, model_name="quicksrnet_medium", scale_factor=4)

    def test_medium_trainer_checkpoint_reload_and_variant_resume_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = build_quicksrnet_from_local_checkpoint(self.medium, model_name="quicksrnet_medium")
            logger = logging.getLogger("capacity-test")
            logger.handlers = [logging.NullHandler()]
            trainer = Trainer(model=model, optimizer=build_optimizer(model, "adam", 1e-4), device=torch.device("cpu"),
                epochs=1, scale_factor=2, config=self.c, run_name="medium-self-check", pretrained_checkpoint=self.medium,
                checkpoint_dir=root / "checkpoints", metric_dir=root / "metrics", sample_dir=root / "samples",
                logger=logger, validation_is_independent=False, save_samples=False)
            items = [{"lr": torch.rand(3, 8, 8), "hr": torch.rand(3, 16, 16), "name": f"synthetic{i}"} for i in range(2)]
            path = trainer.fit(DataLoader(items, batch_size=2), DataLoader(items, batch_size=1))
            self.assertIsNone(trainer.history[-1]["val_ssim"])
            self.assertEqual(verify_checkpoint_reload(checkpoint_path=path, pretrained_checkpoint=self.medium,
                scale_factor=2, trained_model=model, sample_lr=items[0]["lr"].unsqueeze(0), device=torch.device("cpu")), 0)
            trainer.config = self.a
            with self.assertRaisesRegex(ValueError, "variant"):
                trainer.resume(path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA timing check")
    def test_paired_timing_records_same_conditions_and_restores_flags(self):
        flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)
        models = {name: torch.nn.Conv2d(3, 3, 1) for name in ("A", "C")}
        result = benchmark_pair(models, torch.rand(1, 3, 16, 16, device="cuda"), warmup=1, iterations=3, rounds=2)
        self.assertEqual(result["conditions"]["input_shape"], [1, 3, 16, 16])
        self.assertEqual(result["conditions"]["precision"], "FP32, AMP off, TF32 off")
        self.assertEqual(flags, (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark))
        for values in result["models"].values():
            self.assertEqual(len(values["samples_ms"]), 6)
            self.assertGreater(values["median_ms"], 0)

    def test_capacity_plotting_uses_a_c_labels_and_deferred_ssim(self):
        spec = importlib.util.spec_from_file_location("capacity_report", ROOT / "scripts/15_compare_capacity.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        data = {"history": [{"epoch": i, "train_global_l1": .02/i, "val_loss": .02,
                              "val_psnr_db": 26+i/10, "val_edge_region_psnr_db": 20+i/10} for i in range(1, 4)]}
        records = [{"experiment": label, "strategy": strategy, "val_ssim": .86}
                   for label in ("A", "C") for strategy in ("psnr", "weighted", "edge_constrained")]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report.plot_capacity(data, data, records, output)
            self.assertGreater((output / "capacity_training_curves.png").stat().st_size, 1000)
            self.assertGreater((output / "selected_ssim_comparison.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
