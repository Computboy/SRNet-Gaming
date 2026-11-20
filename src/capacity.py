"""Capacity-ablation checks and paired, inference-only GPU timing."""
from __future__ import annotations

import hashlib
import platform
import statistics
import subprocess

import torch

from src.edge_validation import EdgeValidationProtocol
from src.losses import TrainingLossProtocol
from src.metrics import ssim_schedule_from_config

MEDIUM_WEIGHT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/quicksrnet_medium_2x_checkpoint_float32.pth.tar"
MEDIUM_WEIGHT_SHA256 = "a0d176b40a649e45a176c3b53f45e0237015f4f2c17b157ef5c81e38c4442a0d"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_capacity_configs(a, c):
    for field in ("data", "validation", "seed", "device", "output", "dryrun"):
        if a.get(field) != c.get(field):
            raise ValueError(f"A/C must have identical {field}")
    for config in (a, c):
        if TrainingLossProtocol.from_config(config) != TrainingLossProtocol("l1", 0):
            raise ValueError("Both capacity-ablation models must use pure L1 with edge weight 0")
        if ssim_schedule_from_config(config) != "post_selection":
            raise ValueError("SSIM must remain post-selection")
    left, right = dict(a["training"]), dict(c["training"])
    left.pop("edge_l1_weight", None)
    right.pop("edge_l1_weight", None)
    if left != right or right["epochs"] != 100:
        raise ValueError("A/C must share all training parameters and the 100-epoch budget")
    if a["model"]["name"] != "quicksrnet_small" or c["model"]["name"] != "quicksrnet_medium":
        raise ValueError("Expected Small A and Medium C")
    if a["model"]["scale_factor"] != c["model"]["scale_factor"] or c["model"]["scale_factor"] != 2:
        raise ValueError("A/C must both be 2x models")
    if c["model"].get("pretrained_sha256") != MEDIUM_WEIGHT_SHA256:
        raise ValueError("Medium must use the verified official FP32 checkpoint")
    if EdgeValidationProtocol.from_config(a).definition() != EdgeValidationProtocol.from_config(c).definition():
        raise ValueError("A/C edge masks and selection definitions differ")
    return {"shared": ["data", "validation", "seed", "device", "output roots", "dryrun", "training hyperparameters"],
            "changed": ["model: Small f32-m2 -> Medium f32-m5", "corresponding official pretrained weights", "run/output identity"],
            "fairness_limits": ["A resumed after epoch 11 without all RNG/worker states; C is planned as a fresh run",
                                "different official pretrained checkpoints have different pretrained parameter values",
                                "one seed and one game environment do not establish statistical significance"],
            "mask_cache": "No mask cache exists in the baseline; preserve the unchanged HR-only computation"}


def model_description(model):
    convolutions = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    return {"wrapper": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "conv_layers": len(convolutions),
            "conv_shapes": [list(layer.weight.shape) for layer in convolutions]}


def benchmark_pair(models, sample, *, warmup=20, iterations=100, rounds=5):
    """Interleave A/C rounds; measure CUDA forward only on identical input tensors."""
    if sample.device.type != "cuda" or len(models) != 2:
        raise ValueError("Paired timing requires two models and a CUDA input")
    if warmup < 1 or iterations < 1 or rounds < 1:
        raise ValueError("Timing counts must be positive")
    flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.benchmark)
    samples = {name: [] for name in models}
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sample = sample.float()
        for model in models.values():
            model.to(sample.device).float().eval()
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            for model in models.values():
                for _ in range(warmup):
                    model(sample)
            torch.cuda.synchronize()
            names = list(models)
            for round_index in range(rounds):
                order = names if round_index % 2 == 0 else names[::-1]
                for name in order:
                    model = models[name]
                    for _ in range(3):
                        model(sample)
                    torch.cuda.synchronize()
                    events = []
                    for _ in range(iterations):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        model(sample)
                        end.record()
                        events.append((start, end))
                    torch.cuda.synchronize()
                    samples[name].extend(start.elapsed_time(end) for start, end in events)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark = flags
    records = {}
    for name, values in samples.items():
        ordered = sorted(values)
        records[name] = {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                         "p10_ms": ordered[int(.1 * (len(ordered) - 1))],
                         "p90_ms": ordered[int(.9 * (len(ordered) - 1))], "samples_ms": values,
                         "model": model_description(models[name])}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return {"conditions": {"gpu": torch.cuda.get_device_name(sample.device), "torch": str(torch.__version__),
                           "cuda_runtime": torch.version.cuda, "platform": platform.platform(),
                           "input_shape": list(sample.shape), "precision": "FP32, AMP off, TF32 off",
                           "warmup": warmup, "iterations_per_round": iterations, "rounds": rounds,
                           "interleaved_order": "A/C then C/A in alternating rounds",
                           "timing": "CUDA events, model forward only; no transfer, decoding, metrics or saving",
                           "gpu_state_after": gpu.stdout.strip()}, "models": records}
