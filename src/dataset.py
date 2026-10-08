"""Paired LR/HR image dataset for 2x gaming super-resolution.

Real LR frames are read as supplied; this module never replaces them with a
bicubic downsample. To avoid temporal leakage, callers should split continuous
gameplay data by complete scene/sequence before placing it in train, validation,
and test directories. Adjacent frames from one sequence should not be randomly
distributed across those splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImagePair:
    """One validated LR/HR pair."""

    key: str
    lr_path: Path
    hr_path: Path
    lr_size: tuple[int, int]
    hr_size: tuple[int, int]


def _scan_images(directory: Path, side: Literal["LR", "HR"]) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"{side} directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{side} path is not a directory: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix().casefold()
        if key in result:
            raise ValueError(
                f"Ambiguous {side} images share the same relative stem '{key}': "
                f"{result[key]} and {path}"
            )
        result[key] = path

    if not result:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(
            f"No supported images found in {side} directory: {directory}. "
            f"Expected one or more files with suffixes: {supported}"
        )
    return result


def _compatibility_key(key: str, side: Literal["LR", "HR"]) -> str:
    """Normalize the suffixes produced by the existing sanity-check script."""

    suffix = "_lr" if side == "LR" else "_hr"
    return key[: -len(suffix)] if key.endswith(suffix) else key


def _match_paths(lr_dir: Path, hr_dir: Path) -> list[tuple[str, Path, Path]]:
    """Match exact relative stems first, then conventional ``_lr``/``_hr`` stems."""

    lr_images = _scan_images(lr_dir, "LR")
    hr_images = _scan_images(hr_dir, "HR")

    matches: list[tuple[str, Path, Path]] = []
    exact_keys = sorted(lr_images.keys() & hr_images.keys())
    for key in exact_keys:
        matches.append((key, lr_images.pop(key), hr_images.pop(key)))

    def normalized_index(
        images: dict[str, Path], side: Literal["LR", "HR"]
    ) -> dict[str, tuple[str, Path]]:
        index: dict[str, tuple[str, Path]] = {}
        for raw_key, path in images.items():
            key = _compatibility_key(raw_key, side)
            if key in index:
                raise ValueError(
                    f"Ambiguous {side} pairing key '{key}' after removing the "
                    f"conventional _{side.lower()} suffix: {index[key][1]} and {path}"
                )
            index[key] = (raw_key, path)
        return index

    lr_normalized = normalized_index(lr_images, "LR")
    hr_normalized = normalized_index(hr_images, "HR")
    normalized_keys = sorted(lr_normalized.keys() & hr_normalized.keys())
    for key in normalized_keys:
        lr_raw, lr_path = lr_normalized[key]
        hr_raw, hr_path = hr_normalized[key]
        matches.append((key, lr_path, hr_path))
        lr_images.pop(lr_raw)
        hr_images.pop(hr_raw)

    if lr_images or hr_images:
        details: list[str] = []
        if lr_images:
            sample = ", ".join(str(path) for path in list(lr_images.values())[:5])
            details.append(f"Missing HR partners for {len(lr_images)} LR image(s): {sample}")
        if hr_images:
            sample = ", ".join(str(path) for path in list(hr_images.values())[:5])
            details.append(f"Missing LR partners for {len(hr_images)} HR image(s): {sample}")
        raise ValueError("LR/HR pairing failed. " + " | ".join(details))

    return sorted(matches, key=lambda item: item[0])


def _read_size(path: Path, side: Literal["LR", "HR"], key: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to read {side} image for pair '{key}': {path}. "
            f"The file may be damaged or unsupported ({exc})."
        ) from exc
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Invalid {side} image dimensions {size} for pair '{key}': {path}")
    return size


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PairedImageDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Strict paired-image dataset with synchronized training transforms.

    Files are matched recursively by their relative stem (extensions may
    differ). For compatibility with ``02_train_sanity_check.py``, a remaining
    ``*_lr`` file can pair with ``*_hr``. Every image is decoded and its scale
    relationship is checked during construction, so data problems fail before
    the training loop starts.

    Validation should use ``lr_patch_size=None`` and ``augment=False``. Because
    full-resolution validation images can have different shapes, a validation
    DataLoader batch size of 1 is the safest general setting.
    """

    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        *,
        scale_factor: int = 2,
        lr_patch_size: int | None = None,
        augment: bool = False,
    ) -> None:
        self.lr_dir = Path(lr_dir).resolve()
        self.hr_dir = Path(hr_dir).resolve()
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        if lr_patch_size is not None and lr_patch_size < 1:
            raise ValueError(f"lr_patch_size must be >= 1 or None, got {lr_patch_size}")
        self.scale_factor = scale_factor
        self.lr_patch_size = lr_patch_size
        self.augment = augment

        self.pairs: list[ImagePair] = []
        for key, lr_path, hr_path in _match_paths(self.lr_dir, self.hr_dir):
            lr_size = _read_size(lr_path, "LR", key)
            hr_size = _read_size(hr_path, "HR", key)
            expected_hr = (lr_size[0] * scale_factor, lr_size[1] * scale_factor)
            if hr_size != expected_hr:
                raise ValueError(
                    f"Pair '{key}' has LR size {lr_size} and HR size {hr_size}; "
                    f"expected exactly {scale_factor}x HR size {expected_hr}. "
                    f"LR={lr_path}, HR={hr_path}"
                )
            if lr_patch_size is not None and (
                lr_size[0] < lr_patch_size or lr_size[1] < lr_patch_size
            ):
                raise ValueError(
                    f"LR image for pair '{key}' is {lr_size}, smaller than requested "
                    f"{lr_patch_size}x{lr_patch_size} crop: {lr_path}"
                )
            self.pairs.append(ImagePair(key, lr_path, hr_path, lr_size, hr_size))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        try:
            with Image.open(pair.lr_path) as image:
                lr = image.convert("RGB")
            with Image.open(pair.hr_path) as image:
                hr = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Failed to decode pair '{pair.key}' while loading: {exc}") from exc

        if self.lr_patch_size is not None:
            patch = self.lr_patch_size
            max_left = lr.width - patch
            max_top = lr.height - patch
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
            lr = lr.crop((left, top, left + patch, top + patch))
            scale = self.scale_factor
            hr = hr.crop(
                (
                    left * scale,
                    top * scale,
                    (left + patch) * scale,
                    (top + patch) * scale,
                )
            )

        if self.augment:
            if random.random() < 0.5:
                lr = lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                lr = lr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                hr = hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            if rotations:
                angle = rotations * 90
                lr = lr.rotate(angle, expand=True)
                hr = hr.rotate(angle, expand=True)

        return {"lr": _pil_to_tensor(lr), "hr": _pil_to_tensor(hr), "name": pair.key}

