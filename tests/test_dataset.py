from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset


def save_rgb(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    array = np.stack(
        ((xx * 7) % 256, (yy * 11) % 256, ((xx + yy) * 5) % 256), axis=-1
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


class PairedImageDatasetTests(unittest.TestCase):
    def make_dirs(self, base: Path) -> tuple[Path, Path]:
        lr = base / "LR"
        hr = base / "HR"
        lr.mkdir()
        hr.mkdir()
        return lr, hr

    def test_exact_stem_and_sanity_suffix_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lr, hr = self.make_dirs(Path(temporary))
            save_rgb(lr / "scene.png", 16, 12)
            save_rgb(hr / "scene.jpg", 32, 24)
            save_rgb(lr / "legacy_lr.png", 16, 16)
            save_rgb(hr / "legacy_hr.png", 32, 32)
            dataset = PairedImageDataset(lr, hr, scale_factor=2)
            self.assertEqual(len(dataset), 2)
            item = dataset[0]
            self.assertEqual(item["lr"].shape[0], 3)
            self.assertGreaterEqual(float(item["lr"].min()), 0.0)
            self.assertLessEqual(float(item["lr"].max()), 1.0)

    def test_missing_partner_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lr, hr = self.make_dirs(Path(temporary))
            save_rgb(lr / "orphan.png", 8, 8)
            save_rgb(hr / "different.png", 16, 16)
            with self.assertRaisesRegex(ValueError, "Missing HR partners"):
                PairedImageDataset(lr, hr)

    def test_corrupt_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lr, hr = self.make_dirs(Path(temporary))
            (lr / "broken.png").write_bytes(b"not an image")
            save_rgb(hr / "broken.png", 16, 16)
            with self.assertRaisesRegex(ValueError, "Failed to read LR image"):
                PairedImageDataset(lr, hr)

    def test_non_2x_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lr, hr = self.make_dirs(Path(temporary))
            save_rgb(lr / "scene.png", 8, 8)
            save_rgb(hr / "scene.png", 15, 16)
            with self.assertRaisesRegex(ValueError, "expected exactly 2x"):
                PairedImageDataset(lr, hr)

    def test_paired_crop_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lr, hr = self.make_dirs(Path(temporary))
            save_rgb(lr / "scene.png", 20, 18)
            save_rgb(hr / "scene.png", 40, 36)
            dataset = PairedImageDataset(
                lr, hr, scale_factor=2, lr_patch_size=8, augment=True
            )
            item = dataset[0]
            self.assertEqual(tuple(item["lr"].shape), (3, 8, 8))
            self.assertEqual(tuple(item["hr"].shape), (3, 16, 16))


if __name__ == "__main__":
    unittest.main()

