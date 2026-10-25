from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import rgb_psnr_per_image, rgb_ssim


class RGBMetricsTests(unittest.TestCase):
    def test_psnr_identical_is_infinite(self):
        image = torch.rand(2, 3, 16, 17)
        self.assertTrue(torch.isposinf(rgb_psnr_per_image(image, image)).all())

    def test_psnr_known_error_and_rgb_channel_weighting(self):
        target = torch.zeros(2, 3, 16, 16)
        prediction = target.clone()
        prediction[0] = 0.1
        prediction[1, 0] = 1.0
        scores = rgb_psnr_per_image(prediction, target)
        self.assertAlmostEqual(scores[0].item(), 20.0, places=5)
        self.assertAlmostEqual(scores[1].item(), 10 * math.log10(3), places=5)

    def test_psnr_is_exactly_the_legacy_validator(self):
        generator = torch.Generator().manual_seed(99)
        prediction = torch.rand(3, 3, 32, 35, generator=generator) * 1.5 - 0.25
        target = torch.rand(3, 3, 32, 35, generator=generator)
        mse = torch.mean((prediction.clamp(0.0, 1.0) - target) ** 2, dim=(1, 2, 3))
        old = torch.where(mse == 0, torch.full_like(mse, float("inf")), -10.0 * torch.log10(mse))
        self.assertTrue(torch.equal(old, rgb_psnr_per_image(prediction, target)))

    def test_ssim_identity_and_known_rgb_constant(self):
        target = np.zeros((32, 32, 3), dtype=np.float32)
        self.assertEqual(rgb_ssim(target, target), 1.0)
        prediction = target.copy()
        prediction[..., 0] = 0.5
        expected_red_ssim = 0.01**2 / (0.5**2 + 0.01**2)
        self.assertAlmostEqual(rgb_ssim(prediction, target), (2 + expected_red_ssim) / 3, places=6)

    def test_invalid_channel_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            rgb_psnr_per_image(torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
        with self.assertRaises(ValueError):
            rgb_ssim(np.zeros((32, 32)), np.zeros((32, 32)))


if __name__ == "__main__":
    unittest.main()
