import platform
import sys

print("=" * 72)
print("QuickSRNet-Gaming environment check")
print("=" * 72)
print("Python:", sys.version.replace("\n", " "))
print("OS:", platform.platform())

try:
    import torch
except Exception as e:
    print("\n[PyTorch] NOT AVAILABLE")
    print("Reason:", repr(e))
    print("\nNext step: install the CUDA-enabled PyTorch build after checking `nvidia-smi`.")
    raise SystemExit(0)

print("\n[PyTorch]")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        vram_gb = prop.total_memory / (1024 ** 3)
        print(f"GPU {i}: {prop.name}")
        print(f"  VRAM: {vram_gb:.2f} GB")
        print(f"  Compute capability: {prop.major}.{prop.minor}")
else:
    print("\nWARNING: PyTorch cannot currently use CUDA.")
    print("If this computer has an NVIDIA discrete GPU, do not start formal training yet.")
