import torch

from qai_hub_models.models.quicksrnetsmall.model import QuickSRNetSmall

SCALE = 2
H = 128
W = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("QuickSRNet-Small 2x smoke test")
print("=" * 72)
print("Device:", device)

# Official Qualcomm AI Hub wrapper.
# The first run may download the pretrained checkpoint.
model = QuickSRNetSmall.from_pretrained(scale_factor=SCALE).to(device)
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
