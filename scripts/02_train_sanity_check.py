"""QuickSRNet-Small x2: minimal, reproducible fine-tuning sanity check.

Run from any working directory:
    python scripts/02_train_sanity_check.py
    python scripts/02_train_sanity_check.py --steps 200 --lr 1e-4
    python scripts/02_train_sanity_check.py --hr path/to/a/screenshot.png

Precondition:
    Copy the official cached x2 checkpoint to:
    checkpoints/pretrained/quicksrnet_small_2x_checkpoint_float32.pth.tar

This is an *overfitting/pipeline* test on ONE synthetic (or supplied) HR image.
The LR counterpart is made by bicubic downsampling. Results are NOT an
unbiased benchmark of SR quality or generalization to game screenshots.

The Qualcomm implementation exposes a private source-model factory used by
QuickSRNetSmall.from_pretrained(); we use that same factory so loading does not
fetch any remote asset. Pin qai-hub-models if you intend to reproduce results.
"""

"""
1. 读取本地权重：直接从 checkpoints/pretrained/ 加载官方 2× checkpoint。
2. 自动生成模拟数据：如果没有测试图像，就生成一张 256×256 的合成 HR 图，并 Bicubic 下采样为 128×128 LR。
3. 进行 200 步训练：使用 L1 Loss + Adam，执行反向传播与参数更新。
4. 记录训练指标：输出每个 Step 的 L1 Loss 和 PSNR。
5. 保存模型：生成 checkpoints/finetuned/sanity_check.pth。
6. 重新加载模型：验证训练后的参数能正确恢复，并与保存前的推理结果一致。
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from qai_hub_models.models.quicksrnetsmall.model import (
    QuickSRNetSmall,
    _load_quicksrnet_source_model,
)


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = (
    ROOT
    / "checkpoints"
    / "pretrained"
    / "quicksrnet_small_2x_checkpoint_float32.pth.tar"
)
FINETUNED = ROOT / "checkpoints" / "finetuned" / "sanity_check.pth"
DRYRUN_HR = ROOT / "data" / "dryrun" / "HR"
DRYRUN_LR = ROOT / "data" / "dryrun" / "LR"
SAMPLES = ROOT / "outputs" / "samples"
METRICS = ROOT / "outputs" / "metrics"
LOGS = ROOT / "outputs" / "logs"
SCALE = 2
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning sanity check on one paired image")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--hr-size", type=int, default=256, help="Square HR test patch size (even)")
    parser.add_argument("--hr", type=Path, default=None, help="Optional local HR photo/screenshot")
    parser.add_argument("--checkpoint", type=Path, default=PRETRAINED, help="Original Qualcomm .pth.tar")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if args.hr_size < 64 or args.hr_size % SCALE:
        parser.error("--hr-size must be an even integer >= 64")
    return args


def create_demo_hr(size: int) -> Image.Image:
    """Deterministic scene-like colors, hard edges, stripes and fine details."""
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:size, 0:size]
    yyf, xxf = yy / size, xx / size
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[..., 0] = 90 + 50 * (1 - yyf) + 10 * np.sin(10 * xxf)
    rgb[..., 1] = 120 + 55 * (1 - yyf)
    rgb[..., 2] = 145 + 65 * (1 - yyf)
    ground = yyf > 0.65
    rgb[ground, 0] = 100 + 12 * np.sin(65 * xxf[ground])
    rgb[ground, 1] = 109 + 15 * np.sin(70 * xxf[ground])
    rgb[ground, 2] = 78 + 10 * np.sin(70 * xxf[ground])
    noise = rng.normal(0, 2, (size, size, 1))
    arr = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # House, roof, windows: edge reconstruction and geometric patterns.
    draw.rectangle((int(.12*size), int(.30*size), int(.62*size), int(.77*size)), fill=(185, 153, 117))
    draw.polygon([(int(.08*size), int(.31*size)), (int(.37*size), int(.12*size)),
                  (int(.66*size), int(.31*size))], fill=(111, 58, 56))
    for y in range(int(.34*size), int(.75*size), max(7, size // 22)):
        draw.line((int(.12*size), y, int(.62*size), y), fill=(133, 101, 79), width=1)
    for row, y in enumerate(range(int(.38*size), int(.73*size), max(12, size // 12))):
        shift = (size // 17) if row % 2 else 0
        for x in range(int(.14*size)+shift, int(.62*size), max(12, size // 10)):
            draw.line((x, y, x, min(y + size // 18, int(.76*size))), fill=(144, 111, 88), width=1)
    draw.rectangle((int(.22*size), int(.43*size), int(.33*size), int(.61*size)), fill=(36, 60, 84))
    draw.rectangle((int(.43*size), int(.43*size), int(.54*size), int(.61*size)), fill=(36, 60, 84))
    for x in (int(.275*size), int(.485*size)):
        draw.line((x, int(.43*size), x, int(.61*size)), fill=(224, 211, 166), width=2)
    draw.rectangle((int(.32*size), int(.60*size), int(.42*size), int(.77*size)), fill=(99, 57, 43))

    # Trees and thin high-contrast diagonal fences.
    draw.rectangle((int(.80*size), int(.40*size), int(.83*size), int(.78*size)), fill=(87, 58, 37))
    for k in range(18):
        cx = int((.82 + rng.normal(0, .075)) * size)
        cy = int((.36 + rng.normal(0, .09)) * size)
        radius = max(3, size // 23)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                     fill=(int(rng.integers(25, 65)), int(rng.integers(90, 150)), int(rng.integers(35, 70))))
    for offset in range(-size // 4, size, max(6, size // 20)):
        draw.line((offset, size-1, offset + size//5, int(.76*size)), fill=(207, 194, 144), width=2)
    return img
# 没有训练图像时程序化生成一张高分辨率图，再下采样一张LR

def get_paired_images(args: argparse.Namespace) -> tuple[Image.Image, Image.Image, str]:
    for path in (DRYRUN_HR, DRYRUN_LR, SAMPLES, METRICS, LOGS, FINETUNED.parent):
        path.mkdir(parents=True, exist_ok=True)

    if args.hr is not None:
        if not args.hr.is_file():
            raise FileNotFoundError(f"HR image does not exist: {args.hr}")
        source = args.hr.resolve()
    else:
        # This is source material, not an LR/HR pair. Keep it outside the
        # strict paired dataset directories used by 03_finetune.py.
        source = SAMPLES / "sanity_source.png"
        if not source.is_file():
            create_demo_hr(args.hr_size).save(source)
            print(f"Created synthetic HR source: {source}")

    with Image.open(source) as img:
        image = img.convert("RGB")
    if min(image.size) < args.hr_size:
        # Upsampling an undersized HR source is unsuitable for this test.
        raise ValueError(f"HR source {source} is {image.size}; must be >= {args.hr_size}x{args.hr_size}")

    w, h = image.size
    left = (w - args.hr_size) // 2
    top = (h - args.hr_size) // 2
    hr = image.crop((left, top, left + args.hr_size, top + args.hr_size))
    lr = hr.resize((args.hr_size // SCALE, args.hr_size // SCALE), RESAMPLE_BICUBIC)
    hr.save(DRYRUN_HR / "sanity_pair_hr.png")
    lr.save(DRYRUN_LR / "sanity_pair_lr.png")
    return lr, hr, str(source)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(img, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    array = (x.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
    return Image.fromarray(array)


def build_model_from_local_checkpoint(path: Path) -> QuickSRNetSmall:
    if not path.is_file():
        raise FileNotFoundError(
            f"No local pretrained weights: {path}\n"
            "Copy the official 2x .pth.tar from %USERPROFILE%\\.qaihm\\...\\quicksrnetsmall\\v2\\ "
            "into checkpoints/pretrained/ first."
        )
    # Exactly the architecture settings used by the official Small 2x loader.
    backbone = _load_quicksrnet_source_model(SCALE, 32, 2, False)
    # Trusted, official Qualcomm checkpoint ONLY. Their original loader uses
    # weights_only=False as well; do not load arbitrary untrusted pickle files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected official QAI checkpoint dict containing 'state_dict'.")
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    return QuickSRNetSmall(backbone, SCALE)


def score(model: torch.nn.Module, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(lr)
        loss = torch.nn.functional.l1_loss(pred, hr).item()
        mse = torch.mean((pred.clamp(0, 1) - hr) ** 2).item()
        psnr = float("inf") if mse == 0 else 10 * math.log10(1 / mse)
    return pred, loss, psnr


def save_loss_graph(records: list[tuple[int, float, float]], path: Path) -> None:
    """Plot with Pillow so the script doesn't depend on matplotlib."""
    canvas = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = 85, 60, 850, 420
    draw.text((85, 18), "QuickSRNet-Small x2 | sanity training L1 loss", fill=(25, 36, 55))
    draw.line((x0, y0, x0, y1, x1, y1), fill=(40, 45, 50), width=2)
    values = [item[1] for item in records]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * .08, 0.001)
    lo = max(0.0, lo - pad)
    hi += pad
    nsteps = max(1, records[-1][0])
    for i in range(5):
        val = lo + (hi - lo) * (4 - i) / 4
        y = y0 + (y1-y0) * i/4
        draw.line((x0, int(y), x1, int(y)), fill=(230, 235, 242), width=1)
        draw.text((8, int(y)-6), f"{val:.4f}", fill=(65, 75, 88))
    for i in range(5):
        x = x0 + (x1-x0) * i/4
        draw.text((int(x)-8, y1+12), str(int(nsteps * i/4)), fill=(65, 75, 88))
    pts = [
        (int(x0 + (x1-x0)*step/nsteps),
         int(y1 - (y1-y0)*(loss-lo)/(hi-lo)))
        for step, loss, _ in records
    ]
    if len(pts) > 1:
        draw.line(pts, fill=(21, 113, 201), width=3)
    draw.text((430, 459), "Optimizer step", fill=(50, 58, 67))
    canvas.save(path)


def save_comparison(lr: Image.Image, before: Image.Image, after: Image.Image, hr: Image.Image) -> None:
    hr_size = hr.size[0]
    imgs = [lr.resize(hr.size, RESAMPLE_BICUBIC), before, after, hr]
    names = ["LR (bicubic upscaled)", "Before fine-tune", "After fine-tune", "Ground truth HR"]
    strip = Image.new("RGB", (4 * hr_size, hr_size + 36), "white")
    draw = ImageDraw.Draw(strip)
    for i, (img, name) in enumerate(zip(imgs, names)):
        strip.paste(img, (i * hr_size, 36))
        draw.text((i * hr_size + 8, 10), name, fill=(15, 20, 25))
    strip.save(SAMPLES / "sanity_comparison.png")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)

    print("=" * 72)
    print("QuickSRNet-Small x2 - fine-tuning sanity check")
    print("=" * 72)
    print(f"Project: {ROOT}\nDevice: {device}\nCheckpoint: {args.checkpoint}")
    lr_img, hr_img, source = get_paired_images(args)
    x, y = pil_to_tensor(lr_img, device), pil_to_tensor(hr_img, device)
    print(f"Source: {source}\nLR: {tuple(x.shape)}  HR: {tuple(y.shape)}")

    model = build_model_from_local_checkpoint(args.checkpoint).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {nparam:,}")
    if nparam != 22860:
        print("WARNING: parameter count differs from your previous 22,860; check package versions")

    # Snapshot one parameter to prove that the optimizer really updates weights.
    first_parameter = next(model.parameters())
    start_weights = first_parameter.detach().cpu().clone()
    before_tensor, before_loss, before_psnr = score(model, x, y)
    tensor_to_pil(before_tensor).save(SAMPLES / "sanity_before.png")
    print(f"[Step 0000] train-pair L1: {before_loss:.6f}, PSNR: {before_psnr:.2f} dB")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    records: list[tuple[int, float, float]] = [(0, before_loss, before_psnr)]
    interval = max(1, args.steps // 10)

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        if output.shape != y.shape:
            raise RuntimeError(f"Output shape {output.shape} != target {y.shape}")
        loss = torch.nn.functional.l1_loss(output, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("No gradients - check that the model parameters require grad")
        optimizer.step()

        # Record post-update metrics, not the pre-update training loss.
        with torch.no_grad():
            pred = model(x)
            metric_loss = torch.nn.functional.l1_loss(pred, y).item()
            mse = torch.mean((pred.clamp(0, 1) - y) ** 2).item()
            metric_psnr = float("inf") if mse == 0 else 10 * math.log10(1 / mse)
        records.append((step, metric_loss, metric_psnr))
        if step == 1 or step % interval == 0 or step == args.steps:
            print(f"[Step {step:04d}] train-pair L1: {metric_loss:.6f}, PSNR: {metric_psnr:.2f} dB")

    after_tensor, after_loss, after_psnr = score(model, x, y)
    before_img, after_img = tensor_to_pil(before_tensor), tensor_to_pil(after_tensor)
    after_img.save(SAMPLES / "sanity_after.png")
    save_comparison(lr_img, before_img, after_img, hr_img)
    save_loss_graph(records, METRICS / "sanity_loss_curve.png")
    with (METRICS / "sanity_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "l1", "psnr_db"))
        writer.writerows(records)

    # Save raw backbone state_dict (compatible with the source-model loader).
    torch.save(
        {
            "state_dict": model.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": args.steps,
            "scale_factor": SCALE,
            "learning_rate": args.lr,
            "source_pretrained": str(args.checkpoint),
        },
        FINETUNED,
    )
    saved = torch.load(FINETUNED, map_location="cpu", weights_only=True)
    reloaded = build_model_from_local_checkpoint(args.checkpoint).to(device)
    reloaded.model.load_state_dict(saved["state_dict"], strict=True)
    loaded_output, loaded_loss, _ = score(reloaded, x, y)
    max_reload_diff = (loaded_output - after_tensor).abs().max().item()
    max_weight_diff = (first_parameter.detach().cpu() - start_weights).abs().max().item()
    if max_weight_diff <= 0:
        raise AssertionError("Optimizer did not modify the selected parameter")
    if max_reload_diff > 1e-6:
        raise AssertionError(f"Checkpoint reload mismatch (max abs diff={max_reload_diff})")
    if abs(loaded_loss - after_loss) > 1e-6:
        raise AssertionError("Reloaded checkpoint loss differs from trained model loss")

    summary = {
        "status": "PASS",
        "device": str(device),
        "steps": args.steps,
        "lr": args.lr,
        "num_parameters": nparam,
        "source": source,
        "pretrained": str(args.checkpoint),
        "checkpoint": str(FINETUNED),
        "l1_before": before_loss,
        "l1_after": after_loss,
        "psnr_before_db": before_psnr,
        "psnr_after_db": after_psnr,
        "max_parameter_change": max_weight_diff,
        "max_reload_output_difference": max_reload_diff,
        "note": "Training-pair metrics only; NOT independent test performance",
    }
    (LOGS / "sanity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 72)
    print(f"Initial -> final L1:  {before_loss:.6f} -> {after_loss:.6f}")
    print(f"Initial -> final PSNR: {before_psnr:.2f} -> {after_psnr:.2f} dB (training pair)")
    print(f"Max parameter change: {max_weight_diff:.8g}")
    print(f"Max reload output diff: {max_reload_diff:.8g}")
    if after_loss >= before_loss:
        print("WARNING: loss did not decrease; check the learning rate / number of steps")
    print(f"Saved: {FINETUNED}")
    print(f"Samples: {SAMPLES}")
    print(f"Metrics: {METRICS}")
    print("PASS: backward, optimizer update, save, reload and inference completed")


if __name__ == "__main__":
    main()
