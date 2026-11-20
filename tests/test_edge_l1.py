from __future__ import annotations

import copy
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol
from src.losses import EdgeAwareL1, TrainingLossProtocol, edge_region_l1_per_image
from src.trainer import Trainer

spec = importlib.util.spec_from_file_location("edge_l1_reference", Path(__file__).parent / "references/edge_l1_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def config(mode="l1_edge", weight=0.1):
    return {"training": {"loss": mode, "edge_l1_weight": weight},
            "validation": {"edge_evaluation": {"enabled": True}}}


class EdgeL1Tests(unittest.TestCase):
    def objective(self, weight=0.1, device="cpu"):
        return EdgeAwareL1(TrainingLossProtocol("l1_edge", weight),
                           EdgeRegionMetric(EdgeValidationProtocol(enabled=True))).to(device)

    def test_reused_mask_matches_user_reference_pixelwise(self):
        generator = torch.Generator().manual_seed(2026)
        hr = torch.rand(4, 3, 32, 35, generator=generator)
        metric = EdgeRegionMetric(EdgeValidationProtocol(enabled=True))
        self.assertTrue(torch.equal(metric.mask(hr), reference.hr_edge_mask(hr)))
        self.assertTrue(torch.equal(metric.mask(hr.flip(-1)), reference.hr_edge_mask(hr.flip(-1))))

    def test_reference_loss_values_and_gradient_match(self):
        generator = torch.Generator().manual_seed(33)
        hr = torch.rand(2, 3, 20, 24, generator=generator)
        sr = torch.rand(2, 3, 20, 24, generator=generator).requires_grad_()
        ours = self.objective()(sr, hr)
        total, global_l1, edge_l1, coverage = reference.combined_l1_loss(sr, hr, 0.1)
        for a, b in ((ours["total_loss"], total), (ours["global_l1"], global_l1),
                     (ours["edge_l1"], edge_l1), (ours["edge_coverage"], coverage)):
            self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-6))
        grad_a = torch.autograd.grad(ours["total_loss"], sr, retain_graph=True)[0]
        grad_b = torch.autograd.grad(total, sr)[0]
        self.assertTrue(torch.allclose(grad_a, grad_b, atol=1e-9, rtol=1e-6))

    def test_empty_masks_are_zero_and_remain_differentiable(self):
        hr = torch.full((2, 3, 16, 16), 0.5)
        sr = torch.full_like(hr, 0.6, requires_grad=True)
        result = self.objective()(sr, hr)
        self.assertEqual(result["edge_l1"].item(), 0)
        self.assertEqual(result["empty_masks"].item(), 2)
        self.assertEqual(result["edge_coverage"].item(), 0)
        self.assertTrue(torch.equal(result["total_loss"], result["global_l1"]))
        result["total_loss"].backward()
        self.assertTrue(torch.isfinite(sr.grad).all())

    def test_each_image_is_normalized_separately_including_empty_images(self):
        hr = torch.zeros(3, 3, 4, 4)
        sr = torch.ones_like(hr)
        sr[1] *= 0.5
        mask = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
        mask[0, :, 1, 1] = True
        mask[1, :, 1:3, 1:3] = True
        result = edge_region_l1_per_image(sr, hr, mask)
        self.assertTrue(torch.equal(result, torch.tensor([1., 0.5, 0.])))
        self.assertEqual(result.mean().item(), 0.5)

    def test_lambda_zero_is_exact_global_l1_value_and_gradient(self):
        hr = torch.rand(2, 3, 16, 16)
        sr = (torch.rand_like(hr) * 2 - 0.5).requires_grad_()
        result = self.objective(weight=0)(sr, hr)
        original = torch.nn.functional.l1_loss(sr, hr)
        self.assertTrue(torch.equal(result["total_loss"], original))
        a = torch.autograd.grad(result["total_loss"], sr, retain_graph=True)[0]
        b = torch.autograd.grad(original, sr)[0]
        self.assertTrue(torch.equal(a, b))

    def test_training_sr_is_not_clamped_and_invalid_shapes_are_rejected(self):
        hr = torch.zeros(1, 3, 16, 16)
        hr[:, :, :, 8:] = 1
        sr = torch.full_like(hr, 2., requires_grad=True)
        result = self.objective()(sr, hr)
        self.assertGreater(result["global_l1"].item(), 1)
        result["total_loss"].backward()
        self.assertTrue((sr.grad > 0).all())
        with self.assertRaises(ValueError):
            self.objective()(sr[:, :1], hr[:, :1])
        with self.assertRaises(ValueError):
            self.objective()(sr[:, :, :-1], hr)

    def test_invalid_weights_and_loss_modes_are_rejected(self):
        for mode, weight in (("l1", 0.1), ("l1_edge", -1), ("l1_edge", float("nan")),
                             ("l1_edge", float("inf")), ("l1_edge", True), ("unknown", 0)):
            with self.subTest(mode=mode, weight=weight), self.assertRaises(ValueError):
                TrainingLossProtocol.from_config(config(mode, weight))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP test")
    def test_cuda_amp_finite_gradient_and_lambda_zero_parity(self):
        hr = torch.rand(2, 3, 32, 32, device="cuda")
        sr = torch.rand_like(hr).half().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            result = self.objective(device="cuda")(sr, hr)
            zero = self.objective(weight=0, device="cuda")(sr, hr)["total_loss"]
            original = torch.nn.functional.l1_loss(sr, hr)
        self.assertEqual(result["total_loss"].dtype, torch.float32)
        self.assertTrue(torch.equal(zero, original))
        result["total_loss"].backward()
        self.assertTrue(torch.isfinite(sr.grad).all())
        self.assertGreater(sr.grad.abs().sum().item(), 0)


class TinySR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2))

    def forward(self, value):
        return self.model(value)


class EdgeL1IntegrationTests(unittest.TestCase):
    def make_trainer(self, root, settings, model=None, epochs=2):
        model = TinySR() if model is None else model
        logger = logging.getLogger("edge-l1-tests")
        logger.handlers = [logging.NullHandler()]
        return Trainer(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=torch.device("cpu"), epochs=epochs, scale_factor=2, config=settings,
            run_name="edge-l1-test", pretrained_checkpoint=root / "unused", checkpoint_dir=root / "checkpoints",
            metric_dir=root / "metrics", sample_dir=root / "samples", logger=logger,
            validation_is_independent=True, save_samples=False)

    def loader(self):
        generator = torch.Generator().manual_seed(42)
        return DataLoader([{"lr": torch.rand(3, 8, 8, generator=generator),
                            "hr": torch.rand(3, 16, 16, generator=generator), "name": f"sample{i}"}
                           for i in range(2)], batch_size=1)

    def test_zero_weight_training_matches_l1_parameters_exactly(self):
        torch.manual_seed(71)
        initial = TinySR()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.make_trainer(root / "a", config("l1", 0), copy.deepcopy(initial))
            zero = self.make_trainer(root / "b", config("l1_edge", 0), copy.deepcopy(initial))
            baseline.fit(self.loader(), self.loader())
            zero.fit(self.loader(), self.loader())
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline.model.parameters(), zero.model.parameters())))
            self.assertEqual([r["train_loss"] for r in baseline.history], [r["train_loss"] for r in zero.history])

    def test_resume_locks_weight_and_keeps_selection_and_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_trainer(root, config(), epochs=1)
            first.fit(self.loader(), self.loader())
            last = first.checkpoint_dir / "last.pth"
            for mode, weight in (("l1_edge", 0.2), ("l1", 0)):
                changed = self.make_trainer(root, config(mode, weight))
                with self.assertRaisesRegex(ValueError, "Loss"):
                    changed.resume(last)
            resumed = self.make_trainer(root, config())
            resumed.fit(self.loader(), self.loader(), start_epoch=resumed.resume(last))
            self.assertEqual(len(resumed.history), 2)
            row = resumed.history[-1]
            self.assertAlmostEqual(row["train_total_loss"], row["train_global_l1"] + 0.1 * row["train_edge_l1"], places=6)
            for name in ("best_psnr.pth", "best_weighted.pth", "best_edge_constrained.pth"):
                self.assertTrue((resumed.checkpoint_dir / name).is_file())
            self.assertIn("training_loss_protocol", torch.load(last, weights_only=True))


if __name__ == "__main__":
    unittest.main()
