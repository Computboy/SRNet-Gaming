"""Reference implementation for Stage 2 QuickSRNet edge-aware L1.

Not integrated into the user's Windows project. Prefer reusing the exact edge-mask
function from the Stage 1 validation metrics to avoid definition drift.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def hr_edge_mask(
    hr: torch.Tensor,
    threshold: float = 0.05,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """Return a boolean mask [B,1,H,W] derived only from HR RGB [0,1].

    Matches the procedure documented in edge_selection_stage1_zh.md:
    RGB-encoded luma -> 3x3 Sobel /8, valid convolution -> magnitude threshold
    -> radius-1 square dilation -> exclude the outer one-pixel image border.

    Important: this is a reference implementation; verify exact equivalence
    against the existing Stage 1 code before integration.
    """
    if hr.ndim != 4 or hr.shape[1] != 3:
        raise ValueError(f"Expected HR of shape [B,3,H,W], got {tuple(hr.shape)}")
    if hr.shape[-2] < 3 or hr.shape[-1] < 3:
        raise ValueError("HR spatial dimensions must be at least 3x3")
    if dilation_radius != 1:
        raise ValueError("Stage 1 protocol fixes dilation_radius=1")
    if threshold != 0.05:
        raise ValueError("Stage 1 protocol fixes threshold=0.05")

    rgb = hr.detach().float()
    y = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2]
         + 0.114 * rgb[:, 2:3])
    kx = rgb.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = rgb.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(y, kx, padding=0)
    gy = F.conv2d(y, ky, padding=0)
    interior_edges = torch.sqrt(gx.square() + gy.square()) >= threshold

    edges = F.pad(interior_edges.float(), (1, 1, 1, 1))
    mask = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1) > 0
    mask[:, :, 0, :] = False
    mask[:, :, -1, :] = False
    mask[:, :, :, 0] = False
    mask[:, :, :, -1] = False
    return mask


def edge_region_l1(sr: torch.Tensor, hr: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Mean RGB absolute error per masked HR pixel, averaged per image.

    Empty masks contribute zero, rather than NaN or a falsely good metric.
    Assumes SR/HR [B,3,H,W] and mask [B,1,H,W].
    """
    if sr.shape != hr.shape or sr.ndim != 4 or sr.shape[1] != 3:
        raise ValueError("SR/HR must have matching shape [B,3,H,W]")
    if mask.shape != (sr.shape[0], 1, sr.shape[2], sr.shape[3]):
        raise ValueError("Mask must have shape [B,1,H,W]")

    diff = (sr.float() - hr.float()).abs()
    mask_f = mask.to(dtype=diff.dtype)
    per_image_sum = (diff * mask_f).sum(dim=(1, 2, 3))
    per_image_den = mask_f.sum(dim=(1, 2, 3)) * 3.0
    per_image = torch.where(per_image_den > 0,
                            per_image_sum / per_image_den.clamp_min(1.0),
                            torch.zeros_like(per_image_sum))
    return per_image.mean()


def combined_l1_loss(
    sr: torch.Tensor,
    hr: torch.Tensor,
    edge_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, global_l1, edge_l1, edge_coverage).

    Use the project's original global L1 implementation when integrating; it
    should be left unchanged. This reference uses F.l1_loss by default.
    """
    if edge_weight < 0:
        raise ValueError("edge_weight must be nonnegative")
    global_l1 = F.l1_loss(sr, hr)
    mask = hr_edge_mask(hr)
    edge_l1 = edge_region_l1(sr, hr, mask)
    total = global_l1 + edge_weight * edge_l1
    coverage = mask.float().mean()
    return total, global_l1, edge_l1, coverage
