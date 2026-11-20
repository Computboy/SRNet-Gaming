"""Validation-only HR edge regions and experimental checkpoint selection.

No function in this module changes the training objective. Edge-region PSNR is
RGB pixel fidelity in a fixed HR-defined region, not a direct aliasing score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class EdgeValidationProtocol:
    enabled: bool = False
    sobel_threshold: float = 0.05
    dilation_radius: int = 1
    weighted_alpha: float = 0.3
    psnr_tolerance_db: float = 0.1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EdgeValidationProtocol":
        values = config.get("validation", {}).get("edge_evaluation", {})
        if not isinstance(values, dict):
            raise ValueError("validation.edge_evaluation must be a mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown edge_evaluation settings: {sorted(unknown)}")
        result = cls(**values)
        if not isinstance(result.enabled, bool):
            raise ValueError("edge_evaluation.enabled must be a boolean")
        for name in ("sobel_threshold", "weighted_alpha", "psnr_tolerance_db"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"edge_evaluation.{name} must be a finite number")
        if result.sobel_threshold <= 0:
            raise ValueError("edge_evaluation.sobel_threshold must be > 0")
        if not 0 <= result.weighted_alpha <= 1:
            raise ValueError("edge_evaluation.weighted_alpha must be in [0,1]")
        if result.psnr_tolerance_db < 0:
            raise ValueError("edge_evaluation.psnr_tolerance_db must be >= 0")
        if isinstance(result.dilation_radius, bool) or not isinstance(result.dilation_radius, int) or result.dilation_radius < 0:
            raise ValueError("edge_evaluation.dilation_radius must be an integer >= 0")
        return result

    def definition(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "version": 1,
            "metric": "edge_region_rgb_psnr",
            "mask_source": "HR only",
            "luma": "0.299*R + 0.587*G + 0.114*B; RGB [0,1], no linearization",
            "gradient": "Sobel Gx/Gy divided by 8; Euclidean magnitude; >= threshold",
            "boundary": "valid Sobel; dilate within interior; outermost image border excluded",
            "dilation": "square max-pool, width=2*radius+1 HR pixels",
            "error": "mean RGB squared error over mask pixels, prediction clamped [0,1]",
            "aggregation": "per-image dB then image mean; empty-mask images excluded from edge mean",
            "zero_error": "+infinity; no epsilon",
            "selection": "validation only; ties: higher PSNR then earlier epoch",
        }


class EdgeRegionMetric(torch.nn.Module):
    def __init__(self, protocol: EdgeValidationProtocol) -> None:
        super().__init__()
        self.protocol = protocol
        horizontal = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("sobel", torch.stack((horizontal, horizontal.T)).unsqueeze(1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1))

    @torch.no_grad()
    def mask(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 4 or target.shape[1] != 3 or min(target.shape[-2:]) < 3:
            raise ValueError("Edge mask expects [B,3,H,W] with H,W >= 3")
        with torch.autocast(device_type=target.device.type, enabled=False):
            gray = (target.float() * self.luma).sum(dim=1, keepdim=True)
            gradient = F.conv2d(gray, self.sobel)
            magnitude = gradient.square().sum(dim=1, keepdim=True).sqrt()
            mask = (magnitude >= self.protocol.sobel_threshold).float()
            radius = self.protocol.dilation_radius
            if radius:
                mask = F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)
        return F.pad(mask.bool(), (1, 1, 1, 1), value=False)

    @torch.no_grad()
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        return masked_rgb_psnr_per_image(output, target, self.mask(target))


def masked_rgb_psnr_per_image(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    if output.ndim != 4 or output.shape != target.shape or output.shape[1] != 3:
        raise ValueError("Masked PSNR expects matching [B,3,H,W] tensors")
    if mask.shape != (output.shape[0], 1, *output.shape[-2:]) or mask.dtype != torch.bool:
        raise ValueError("Edge mask must be boolean [B,1,H,W]")
    squared_error = (output.float().clamp(0, 1) - target.float()).square().mean(dim=1, keepdim=True)
    pixels = mask.sum(dim=(1, 2, 3))
    mse = (squared_error * mask).sum(dim=(1, 2, 3)) / pixels.clamp_min(1)
    psnr = -10 * torch.log10(mse)
    psnr = torch.where(pixels > 0, psnr, torch.full_like(psnr, float("nan")))
    return {"psnr_db": psnr, "pixels": pixels, "coverage": mask.float().mean(dim=(1, 2, 3))}


def weighted_score(psnr: float, edge_psnr: float | None, alpha: float) -> float | None:
    if alpha == 0:
        return psnr
    if edge_psnr is None:
        return None
    if alpha == 1:
        return edge_psnr
    return (1 - alpha) * psnr + alpha * edge_psnr


def select_epochs(history: list[dict[str, Any]], protocol: EdgeValidationProtocol) -> dict[str, dict[str, Any]]:
    """Recompute the constrained choice against the CURRENT global PSNR maximum.

    All epoch checkpoints are retained when enabled: an earlier candidate can
    become the winner after a previously selected candidate leaves the window.
    """
    if not history:
        return {}
    key_psnr = lambda row: (float(row["val_psnr_db"]), -int(row["epoch"]))
    best_psnr = max(history, key=key_psnr)
    choices = {"psnr": best_psnr}
    scored = [row for row in history if row.get("val_weighted_score") is not None]
    if scored:
        choices["weighted"] = max(scored, key=lambda row: (
            float(row["val_weighted_score"]), float(row["val_psnr_db"]), -int(row["epoch"])
        ))
    candidates = [row for row in history
                  if row.get("val_edge_region_psnr_db") is not None
                  and float(row["val_psnr_db"]) >= float(best_psnr["val_psnr_db"]) - protocol.psnr_tolerance_db]
    if candidates:
        choices["edge_constrained"] = max(candidates, key=lambda row: (
            float(row["val_edge_region_psnr_db"]), float(row["val_psnr_db"]), -int(row["epoch"])
        ))
    return choices


def json_safe(value: Any) -> Any:
    """Keep JSON standards-compliant for exact reconstruction and empty masks."""
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value == math.inf else "-inf" if value == -math.inf else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value
