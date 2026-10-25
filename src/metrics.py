"""Shared whole-image RGB metrics for training validation and verification."""
from __future__ import annotations

import numpy as np
import torch


SSIM_PARAMETERS = {
    "data_range": 1.0,
    "channel_axis": -1,
    "win_size": 11,
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
    "K1": 0.01,
    "K2": 0.03,
}


def rgb_psnr_per_image(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Original Trainer.validate formula, returning one PSNR per BCHW image.

    Peak=1, RGB channels equally weighted, no spatial border shave, no epsilon.
    Predictions are clamped to [0,1], exactly as in the original validator.
    """
    if output.ndim != 4 or output.shape != target.shape or output.shape[1] != 3:
        raise ValueError("PSNR expects matching [B,3,H,W] prediction and target tensors")
    per_image_mse = torch.mean((output.clamp(0.0, 1.0) - target) ** 2, dim=(1, 2, 3))
    return torch.where(
        per_image_mse == 0,
        torch.full_like(per_image_mse, float("inf")),
        -10.0 * torch.log10(per_image_mse),
    )


def rgb_ssim(output: np.ndarray, target: np.ndarray) -> float:
    """scikit-image RGB SSIM on complete HWC float32 images in [0,1].

    Channels are scored separately and equally averaged. The library's scalar
    SSIM uses the valid window centres (5-pixel map margin for an 11x11 window),
    with reflected Gaussian filtering at image boundaries. No input crop is made.
    """
    from skimage.metrics import structural_similarity

    if output.ndim != 3 or output.shape != target.shape or output.shape[-1] != 3:
        raise ValueError("SSIM expects matching [H,W,3] arrays")
    if min(output.shape[:2]) < SSIM_PARAMETERS["win_size"]:
        raise ValueError("Image is smaller than the SSIM window")
    return float(structural_similarity(target, output, **SSIM_PARAMETERS))
