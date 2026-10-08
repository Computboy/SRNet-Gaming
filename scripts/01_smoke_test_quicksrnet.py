import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer import build_quicksrnet_small_from_local_checkpoint

SCALE = 2
H = 128
W = 128
PRETRAINED = (
    ROOT
    / "checkpoints"
    / "pretrained"
    / "quicksrnet_small_2x_checkpoint_float32.pth.tar"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("QuickSRNet-Small 2x smoke test")
print("=" * 72)
print("Device:", device)

# Use the project-local official checkpoint. This avoids a network request and
# the global QAI cache, while retaining the same official wrapper/architecture.
model = build_quicksrnet_small_from_local_checkpoint(
    PRETRAINED, scale_factor=SCALE
).to(device)
model.eval()

x = torch.rand(1, 3, H, W, device=device)

with torch.no_grad():
    y = model(x)

print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Output min/max:", float(y.min()), float(y.max()))

expected = (1, 3, H * SCALE, W * SCALE)
assert tuple(y.shape) == expected, f"Unexpected output shape: {tuple(y.shape)}, expected {expected}"
assert torch.isfinite(y).all(), "Output contains NaN or Inf."

params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Parameters:", f"{params:,}")
print("Trainable parameters:", f"{trainable:,}")
print("\nPASS: pretrained QuickSRNet-Small 2x loaded and inference succeeded.")
