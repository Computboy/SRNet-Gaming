"""Small configuration, reproducibility, logging, and dry-run helpers."""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_run_name(prefix: str) -> str:
    safe_prefix = "".join(char if char.isalnum() or char in "-_" else "-" for char in prefix)
    return f"{safe_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def create_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _has_image(directory: Path) -> bool:
    suffixes = {".png", ".jpg", ".jpeg"}
    return directory.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes for path in directory.rglob("*")
    )


def _create_demo_hr(size: int = 256) -> Image.Image:
    """Create a deterministic image with gradients, edges, and fine patterns."""

    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx / max(size - 1, 1)
    y = yy / max(size - 1, 1)
    rgb = np.empty((size, size, 3), dtype=np.float32)
    rgb[..., 0] = 50 + 120 * x
    rgb[..., 1] = 75 + 110 * (1 - y)
    rgb[..., 2] = 115 + 60 * np.sin(x * np.pi)
    rgb += rng.normal(0, 1.5, (size, size, 1))
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 60, 150, 205), fill=(186, 148, 105), outline=(35, 32, 30), width=3)
    draw.polygon(((14, 64), (87, 16), (163, 64)), fill=(116, 51, 47))
    draw.rectangle((58, 102, 104, 163), fill=(35, 70, 104), outline=(225, 219, 182), width=3)
    for offset in range(-50, size, 13):
        draw.line((offset, size - 1, offset + 65, 190), fill=(232, 218, 155), width=2)
    for row in range(12, size, 16):
        draw.line((175, row, 246, row), fill=(30, 65 + row // 4, 45), width=2)
    return image


def ensure_dryrun_pair(lr_dir: Path, hr_dir: Path, scale_factor: int) -> bool:
    """Generate a synthetic pair only when both dry-run folders have no images.

    Returns True when files were created. A partially populated pair is left
    untouched so the strict dataset error identifies the data problem.
    """

    lr_has_images = _has_image(lr_dir)
    hr_has_images = _has_image(hr_dir)
    if lr_has_images or hr_has_images:
        return False
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr_size = 256
    if hr_size % scale_factor:
        raise ValueError(f"Synthetic HR size {hr_size} is not divisible by scale {scale_factor}")
    hr = _create_demo_hr(hr_size)
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    lr = hr.resize((hr_size // scale_factor, hr_size // scale_factor), resampling)
    # Matching names exercise the same pairing rule expected from formal data.
    hr.save(hr_dir / "synthetic_scene.png")
    lr.save(lr_dir / "synthetic_scene.png")
    return True

