"""Global L1 with an optional, HR-defined edge-region L1 training term."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn import functional as F

from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol


@dataclass(frozen=True)
class TrainingLossProtocol:
    mode: str = "l1"
    edge_l1_weight: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainingLossProtocol":
        settings = config.get("training", {})
        mode = str(settings.get("loss", "l1")).casefold()
        weight = settings.get("edge_l1_weight", 0.0)
        if mode not in {"l1", "l1_edge"}:
            raise ValueError("training.loss must be l1 or l1_edge")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("training.edge_l1_weight must be finite and >= 0")
        if mode == "l1" and weight != 0:
            raise ValueError("Use training.loss=l1_edge when edge_l1_weight is nonzero")
        return cls(mode=mode, edge_l1_weight=float(weight))

    def definition(self, edge_protocol: EdgeValidationProtocol) -> dict[str, Any]:
        return {
            "version": 1, "mode": self.mode, "edge_l1_weight": self.edge_l1_weight,
            "global_l1": "torch.nn.functional.l1_loss(sr,hr), mean BCHW, no SR clamp",
            "edge_l1": "per-image sum(mask*abs(sr-hr))/(3*mask_pixels), then batch mean",
            "empty_mask": "zero edge loss for that image; included in batch mean",
            "precision": "original autocast global L1; float32 mask and edge reduction",
            "mask": {key: value for key, value in edge_protocol.definition().items()
                     if key in {"version", "mask_source", "luma", "gradient", "boundary", "dilation",
                                "sobel_threshold", "dilation_radius"}},
        }


def edge_region_l1_per_image(
    sr: torch.Tensor, hr: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if sr.ndim != 4 or sr.shape != hr.shape or sr.shape[1] != 3:
        raise ValueError("Edge L1 expects matching [B,3,H,W] SR and HR")
    if mask.shape != (sr.shape[0], 1, *sr.shape[-2:]) or mask.dtype != torch.bool:
        raise ValueError("Edge L1 mask must be boolean [B,1,H,W]")
    with torch.autocast(device_type=sr.device.type, enabled=False):
        difference = (sr.float() - hr.float()).abs()
        pixels = mask.sum(dim=(1, 2, 3))
        error = (difference * mask).sum(dim=(1, 2, 3))
        return error / (3 * pixels.clamp_min(1))


class EdgeAwareL1(torch.nn.Module):
    def __init__(
        self, protocol: TrainingLossProtocol, mask_builder: EdgeRegionMetric | None
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.mask_builder = mask_builder
        if protocol.mode == "l1_edge" and mask_builder is None:
            raise ValueError("l1_edge requires an HR edge mask builder")

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> dict[str, torch.Tensor]:
        if sr.ndim != 4 or sr.shape != hr.shape or sr.shape[1] != 3:
            raise ValueError("L1 training expects matching [B,3,H,W] SR and HR")
        # Preserve the original global-loss operation and training output range.
        global_l1 = F.l1_loss(sr, hr)
        edge_l1 = global_l1.new_zeros(())
        coverage = global_l1.new_zeros(())
        empty = torch.zeros((), dtype=torch.int64, device=sr.device)
        if self.mask_builder is not None:
            mask = self.mask_builder.mask(hr)
            # With lambda=0 this is a diagnostic only; no extra gradient graph.
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.protocol.edge_l1_weight > 0):
                edge_l1 = edge_region_l1_per_image(sr, hr, mask).mean()
            coverage = mask.float().mean()
            empty = (mask.sum(dim=(1, 2, 3)) == 0).sum()
        total = global_l1 if self.protocol.edge_l1_weight == 0 else global_l1 + self.protocol.edge_l1_weight * edge_l1
        return {"total_loss": total, "global_l1": global_l1, "edge_l1": edge_l1,
                "edge_coverage": coverage, "empty_masks": empty}
