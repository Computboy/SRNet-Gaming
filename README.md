# QuickSRNet-Gaming

Starter project for **Gaming-domain fine-tuning of QuickSRNet-Small (2×)**.

## Current phase

This starter intentionally stops before formal training.

The first goal is to verify, in order:

1. Windows/Python environment
2. CUDA-visible NVIDIA GPU
3. Official QuickSRNet-Small pretrained model can be loaded
4. 2× inference works
5. Dry-run LR/HR image pairs can be generated

Only after these checks should `Dataset`, validation metrics, and the fine-tuning loop be added.

## Recommended local path

`F:\CADCG\SRTP\QuickSRNet-Gaming`

## Folder layout

- `configs/`: experiment configuration
- `scripts/`: executable checks and utilities
- `src/`: reusable source code (added in the next phase)
- `data/dryrun/`: temporary images for pipeline testing
- `data/gaming/`: formal UE train/val/test pairs
- `checkpoints/`: pretrained and fine-tuned weights
- `outputs/`: logs, samples, metrics

## First commands

Open **Anaconda Prompt** or PowerShell:

```powershell
cd F:\CADCG\SRTP\QuickSRNet-Gaming
conda create -n qsr-gaming python=3.11 -y
conda activate qsr-gaming
python -m pip install --upgrade pip
```

Before installing PyTorch, run:

```powershell
nvidia-smi
```

Record the GPU model and VRAM. Then install a CUDA-enabled PyTorch build appropriate for the machine.

After PyTorch is installed:

```powershell
python scripts\00_check_env.py
pip install -r requirements.txt
python scripts\01_smoke_test_quicksrnet.py
```

For a dry-run dataset, copy several images into:

`data\dryrun\HR`

then run:

```powershell
python scripts\02_make_dryrun_pairs.py
```

## Important

QuickSRNet expects RGB input values in `[0, 1]`.
Do not use ImageNet mean/std normalization for this model.
