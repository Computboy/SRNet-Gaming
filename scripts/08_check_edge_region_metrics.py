"""Check fixed edge-region metrics on five predeclared VALIDATION frames only.

Compare existing models and controlled blur/shift/block perturbations. No training,
test access, parameter search, or claim of perceptual validation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset
from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol, json_safe, weighted_score
from src.metrics import rgb_psnr_per_image, rgb_ssim
from src.trainer import build_quicksrnet_small_from_local_checkpoint, resolve_device
from src.utils import load_yaml_config, project_path


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blur(image, sigma):
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    result = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"),
                      kernel.reshape(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(result, (0, 0, radius, radius), mode="reflect"),
                    kernel.reshape(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)


def pil_image(tensor):
    array = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/asian_village_selection.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetuned/asian-village-20261018-200240/best.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    out = project_path(ROOT, args.output_dir)
    if out.exists():
        raise ValueError("Choose a new diagnostic output directory")
    config_path = project_path(ROOT, args.config)
    config = load_yaml_config(config_path)
    protocol = EdgeValidationProtocol.from_config(config)
    if not protocol.enabled:
        raise ValueError("Enable validation.edge_evaluation in the diagnostic config")
    data = config["data"]
    # Limit this diagnostic to the established validation split, never an arbitrary test path.
    for name, expected in (("val_lr", "data/gaming/val/LR"), ("val_hr", "data/gaming/val/HR")):
        if project_path(ROOT, data[name]) != project_path(ROOT, expected):
            raise ValueError("This fixed diagnostic only supports the established validation directories")
    dataset = PairedImageDataset(project_path(ROOT, data["val_lr"]), project_path(ROOT, data["val_hr"]))
    if len(dataset) != 240 or any(not pair.key.startswith("scene_1_02/") for pair in dataset.pairs):
        raise ValueError("Expected only the 240-frame Scene_1_02 validation sequence")
    plan_path = ROOT / "outputs/verification/val-scene1-02-20261019-140251/sample_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frames = [0, 60, 120, 180, 239]
    if plan["frames"] != frames:
        raise ValueError("Historical fixed frame plan changed")
    pretrained = project_path(ROOT, config["model"]["pretrained_checkpoint"])
    checkpoint_path = project_path(ROOT, args.checkpoint)
    protected = [config_path, plan_path, pretrained, checkpoint_path]
    protected += [path for i in frames for path in (dataset.pairs[i].lr_path, dataset.pairs[i].hr_path)]
    before = {str(path): fingerprint(path) for path in protected}
    device = resolve_device(args.device)
    models = {name: build_quicksrnet_small_from_local_checkpoint(pretrained).to(device).float().eval().requires_grad_(False)
              for name in ("original", "finetuned")}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    models["finetuned"].model.load_state_dict(checkpoint["state_dict"], strict=True)
    metric = EdgeRegionMetric(protocol).to(device)
    out.mkdir(parents=True)
    (out / "samples").mkdir()
    (out / "sample_plan.json").write_text(json.dumps({"frames": frames, "rois": plan["rois"],
        "fixed_before_scoring": True, "protocol": protocol.definition()}, indent=2), encoding="utf-8")
    rows = []
    for frame in frames:
        item = dataset[frame]
        lr, hr = item["lr"].unsqueeze(0).to(device), item["hr"].unsqueeze(0).to(device)
        with torch.inference_mode():
            predictions = {"bicubic": F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=False)}
            predictions.update({name: model(lr) for name, model in models.items()})
            predictions = {name: tensor.clamp(0, 1) for name, tensor in predictions.items()}
            predictions.update({"hr_identity": hr, "hr_blur_sigma0.5": blur(hr, 0.5),
                "hr_blur_sigma1.0": blur(hr, 1.0), "hr_blur_sigma2.0": blur(hr, 2.0),
                "hr_shift_right1px": F.pad(hr[:, :, :, :-1], (1, 0, 0, 0), mode="replicate"),
                "hr_block2x": F.interpolate(F.interpolate(hr, scale_factor=0.5, mode="nearest"), size=hr.shape[-2:], mode="nearest"),
                "finetuned_blur_sigma0.5": blur(predictions["finetuned"], 0.5)})
            mask = metric.mask(hr)
            results = {}
            reference = hr[0].permute(1, 2, 0).cpu().numpy()
            for name, prediction in predictions.items():
                edge = metric(prediction, hr)
                psnr = rgb_psnr_per_image(prediction, hr).item()
                edge_psnr = edge["psnr_db"].item() if edge["pixels"].item() else None
                result = {"frame": frame, "key": item["name"], "method": name, "psnr_db": psnr,
                    "ssim": rgb_ssim(prediction[0].permute(1, 2, 0).cpu().numpy(), reference),
                    "edge_region_psnr_db": edge_psnr, "edge_coverage": edge["coverage"].item(),
                    "weighted_score": weighted_score(psnr, edge_psnr, protocol.weighted_alpha)}
                rows.append(result)
                results[name] = result
            pil_image(mask.float().expand(-1, 3, -1, -1)).save(out / "samples" / f"frame{frame:04d}_mask.png")
            overlay = hr.clone()
            red = torch.tensor([1., 0.15, 0.1], device=device).reshape(1, 3, 1, 1)
            overlay = torch.where(mask, 0.55 * overlay + 0.45 * red, overlay)
            pil_image(overlay).save(out / "samples" / f"frame{frame:04d}_mask_overlay.png")
            views = ["hr_identity", "bicubic", "original", "finetuned", "finetuned_blur_sigma0.5", "hr_shift_right1px"]
            for region, (label, box) in enumerate(plan["rois"][str(frame)]):
                canvas = Image.new("RGB", (len(views) * 280, 342), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 5), f"Frame {frame:04d} | {label} | Scores below are FULL-FRAME, not ROI", fill="#18212b", font=font(17))
                for column, name in enumerate(views):
                    score = results[name]
                    x = column * 280
                    draw.text((x + 8, 33), name, fill="#18212b", font=font(15))
                    draw.text((x + 8, 55), f"P {score['psnr_db']:.3f} | Edge {score['edge_region_psnr_db']:.3f}", fill="#44515f", font=font(15))
                    canvas.paste(pil_image(predictions[name]).crop(box), (x + 8, 80))
                canvas.save(out / "samples" / f"frame{frame:04d}_roi_{region}.png")
        print(f"Checked validation frame {frame:04d}", flush=True)
    with (out / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = []
    for name in predictions:
        items = [row for row in rows if row["method"] == name]
        means.append({"method": name, **{key: sum(row[key] for row in items) / len(items)
                      for key in ("psnr_db", "ssim", "edge_region_psnr_db", "weighted_score")}})
    after = {str(path): fingerprint(path) for path in protected}
    if before != after:
        raise AssertionError("An input/checkpoint/config changed during the diagnostic")
    summary = {"frames": frames, "sequence": "Scene_1_02", "protocol": protocol.definition(),
        "finetuned_checkpoint_epoch": checkpoint["epoch"], "means": means,
        "input_sha256": before, "protected_files_unchanged": True,
        "training_performed": False, "test_set_used": False,
        "limitation": "Five validation frames; synthetic controls are not a perceptual aliasing benchmark"}
    (out / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = ["# Edge-region PSNR 指标检查", "", "仅使用事先固定的 5 个验证帧；未训练、未访问测试集、未搜索阈值或权重。", "",
        "| 方法 | PSNR (dB) | SSIM | Edge-region PSNR (dB) | 加权分数 |", "|---|---:|---:|---:|---:|"]
    report += [f"| {row['method']} | {row['psnr_db']:.4f} | {row['ssim']:.4f} | {row['edge_region_psnr_db']:.4f} | {row['weighted_score']:.4f} |" for row in means]
    report += ["", "HR 派生的模糊、位移和块状版本用于检查指标响应，不是超分方法。", "",
        "Edge-region PSNR 衡量固定 HR 边缘带内的像素误差，不能独立区分锯齿、模糊或错位。",
        "finetuned_blur_sigma0.5 是固定的输出平滑对照，可用于检查分数是否奖励额外平滑。",
        "图中分数为整帧指标；ROI 只用于目视对比。红色叠加图展示实际计分区域。", "",
        "本检查只能说明指标对这些扰动和模型是否有响应；人眼一致性与选模收益仍待验证。"]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_safe(means), indent=2), flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
