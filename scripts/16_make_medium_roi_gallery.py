"""Create fixed multi-ROI Medium figures and HR/Small/Medium contact sheets.

Only five predeclared validation frames are inferred; no training or Test access.
Reuses the original Small comparison renderer without modifying it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from src.dataset import PairedImageDataset
from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol
from src.metrics import rgb_psnr_per_image, rgb_ssim
from src.post_selection import verify_manifest
from src.trainer import build_quicksrnet_from_local_checkpoint, resolve_device


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def montage(rows, destination, renderer, title):
    width = max(row.width for row in rows)
    header, gap = 64, 16
    canvas = Image.new("RGB", (width, header + sum(row.height for row in rows) + gap * (len(rows)-1)), "#e7edf2")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, font=renderer.font(23, True), fill="#172033")
    draw.text((12, 38), "Fixed frames 0000 / 0060 / 0120 / 0180 / 0239; identical HR coordinates", font=renderer.font(16), fill="#526071")
    top = header
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + gap
    canvas.save(destination, compress_level=6, dpi=(150, 150))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/capacity-medium-l1-001/analysis/multi_roi_comparison"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError("Choose a new output directory; old figures are never overwritten")
    analysis = ROOT / "outputs/experiments/capacity-medium-l1-001/analysis"
    selected = read_csv(analysis / "selected_models.csv")
    choice = next(row for row in selected if row["experiment"] == "C" and row["strategy"] == "psnr")
    baseline = next(row for row in selected if row["experiment"] == "A" and row["strategy"] == "psnr")
    checkpoint_path = Path(choice["checkpoint"])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == int(choice["epoch"])
    assert saved["config"]["model"]["name"] == "quicksrnet_medium"
    config = saved["config"]
    pretrained = ROOT / config["model"]["pretrained_checkpoint"]
    assert sha256(pretrained) == config["runtime"]["pretrained_checkpoint_sha256"]
    before = {str(path): sha256(path) for path in (checkpoint_path, pretrained, Path(baseline["checkpoint"]))}
    plan_path = ROOT / "outputs/experiments/edge-l1-ab-002-ssim-posthoc/analysis/sample_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["frames"] == [0, 60, 120, 180, 239]
    spec = importlib.util.spec_from_file_location("small_figure_renderer", ROOT / "scripts/05_verify_validation.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    for frame in plan["frames"]:
        assert [list(box) for _, box in renderer.ROIS[frame]] == [box for _, box in plan["rois"][str(frame)]]
    renderer.LABELS = ("LR (display only)", "Bicubic 2x", "Official Medium", f"Fine-tuned Medium E{saved['epoch']}", "HR Ground Truth")
    dataset = PairedImageDataset(ROOT / config["data"]["val_lr"], ROOT / config["data"]["val_hr"])
    manifest = json.loads((ROOT / "outputs/logs" / saved["run_name"] / "validation_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(dataset, manifest)
    assert len(dataset) == 240
    expected = {row["key"]: row for row in read_csv(analysis / "post_selection_ssim/C" / f"epoch{saved['epoch']:03d}.csv")}
    device = resolve_device(args.device)
    models = {name: build_quicksrnet_from_local_checkpoint(pretrained, model_name="quicksrnet_medium").to(device).float().eval().requires_grad_(False)
              for name in ("original", "finetuned")}
    models["finetuned"].model.load_state_dict(saved["state_dict"], strict=True)
    edge_metric = EdgeRegionMetric(EdgeValidationProtocol.from_config(config)).to(device)
    out.mkdir(parents=True)
    for name in ("regions_native", "regions_zoom2x", "frame_overviews"):
        (out / name).mkdir()
    medium_rows, capacity_rows = {0: [], 1: []}, {0: [], 1: []}
    scores, regions, max_differences = [], [], {"psnr_db": 0., "ssim": 0., "edge_region_psnr_db": 0.}
    with torch.inference_mode():
        for frame in plan["frames"]:
            item = dataset[frame]
            lr, hr = item["lr"].unsqueeze(0).to(device), item["hr"].unsqueeze(0).to(device)
            predictions = {"bicubic": F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=False).clamp(0, 1)}
            predictions.update({name: model(lr).clamp(0, 1) for name, model in models.items()})
            reference = renderer.hwc(hr)
            row = {"key": item["name"], "frame": frame}
            arrays = {}
            for method, prediction in predictions.items():
                arrays[method] = renderer.hwc(prediction)
                values = {"psnr_db": rgb_psnr_per_image(prediction, hr)[0].item(),
                          "ssim": rgb_ssim(arrays[method], reference),
                          "edge_region_psnr_db": edge_metric(prediction, hr)["psnr_db"][0].item()}
                row.update({f"{method}_{key}": value for key, value in values.items()})
                if method == "finetuned":
                    for key, value in values.items():
                        delta = abs(value - float(expected[item["name"]][key]))
                        max_differences[key] = max(max_differences[key], delta)
                        assert delta <= 1e-5, (frame, key, delta)
            scores.append(row)
            hr_image = renderer.image_from_float(reference)
            images = [renderer.image_from_float(renderer.hwc(lr)).resize(hr_image.size, Image.Resampling.NEAREST)]
            images += [renderer.image_from_float(arrays[name]) for name in renderer.METHODS] + [hr_image]
            rois = [(label, tuple(box)) for label, box in plan["rois"][str(frame)]]
            renderer.comparison_strip(images, row, f"Frame {frame:04d} | Medium 2x | ROI 1: architecture, ROI 2: texture", 512, rois=rois).save(out / "frame_overviews" / f"frame{frame:04d}.png")
            for region, (label, box) in enumerate(rois):
                assert box[2] - box[0] == box[3] - box[1] == 256
                assert 0 <= box[0] < box[2] <= hr_image.width and 0 <= box[1] < box[3] <= hr_image.height
                title = f"Frame {frame:04d} | {label} | identical 256x256 HR crop; scores: full frame"
                native = renderer.comparison_strip(images, row, title, 256, crop=box)
                native.save(out / "regions_native" / f"frame{frame:04d}_roi{region}.png")
                zoom = renderer.comparison_strip(images, row, title + " | 2x nearest", 512, crop=box)
                zoom.save(out / "regions_zoom2x" / f"frame{frame:04d}_roi{region}.png")
                medium_rows[region].append(native)
                capacity_path = analysis / "samples" / f"psnr_frame{frame:04d}_roi{region}.png"
                with Image.open(capacity_path) as source:
                    capacity_rows[region].append(source.convert("RGB").copy())
                regions.append({"frame": frame, "label": label, "hr_box": list(box)})
            print(f"Saved frame {frame:04d}: two aligned ROIs, native and 2x zoom", flush=True)
    for region, name in ((0, "architecture"), (1, "paving")):
        montage(medium_rows[region], out / f"medium_{name}_montage.png", renderer, f"Medium | {name} | LR / Bicubic / Official / Fine-tuned / HR")
        montage(capacity_rows[region], out / f"capacity_{name}_montage.png", renderer, f"Capacity | {name} | HR / Small E{baseline['epoch']} / Medium E{saved['epoch']}")
    all_medium = [medium_rows[r][i] for i in range(5) for r in (0, 1)]
    all_capacity = [capacity_rows[r][i] for i in range(5) for r in (0, 1)]
    montage(all_medium, out / "medium_all_regions.png", renderer, "Medium | 10 fixed ROIs | LR / Bicubic / Official / Fine-tuned / HR")
    montage(all_capacity, out / "capacity_all_regions.png", renderer, "Capacity | 10 fixed ROIs | HR / Small / Medium")
    with (out / "five_frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    assert all(sha256(Path(name)) == checksum for name, checksum in before.items())
    summary = {"medium_epoch": saved["epoch"], "small_epoch": int(baseline["epoch"]),
               "strategy": "psnr", "frames": plan["frames"], "regions": regions,
               "original_fixed_plan": str(plan_path), "checkpoint_hashes": before,
               "inference": "full-frame RGB FP32, batch=1, eval/inference mode; no augmentation or tiling",
               "zoom": "same 256x256 HR coordinates; 2x nearest for all columns",
               "LR": "nearest-neighbour display only; not scored",
               "scores": "complete-frame floats before PNG quantization, not ROI scores or complete-240-frame aggregate",
               "maximum_difference_from_existing_C_scores": max_differences,
               "existing_figures_and_weights_unchanged": True, "training_performed": False, "test_accessed": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# Medium 多帧局部对比图", "",
            "使用原先固定的验证帧 0、60、120、180、239，每帧两个 256×256 HR 区域；没有重新挑选表现最好的区域。", "",
            "五列 Medium 图从左到右：LR 最近邻展示、Bicubic、官方 Medium、微调 Medium（第 98 轮）、HR。",
            "三列容量图从左到右：HR、Small + L1（第 100 轮）、Medium + L1（第 98 轮）。均按原最高 PSNR 规则选模。", "",
            "每列使用相同裁剪坐标；单独放大图统一采用 2 倍最近邻。图上 PSNR/SSIM 是整帧分数，不是局部 ROI 分数。",
            "本次只推理固定五帧供展示；完整 240 帧结果继续以原容量消融报告为准。Edge-region PSNR 同时记录在 five_frame_metrics.csv。", "",
            "## 五列 Medium 对照", "", "![建筑轮廓](medium_architecture_montage.png)", "", "![地面纹理](medium_paving_montage.png)", "",
            "[全部 10 个区域合图](medium_all_regions.png)", "", "## 三列容量对照", "",
            "![建筑轮廓容量对照](capacity_architecture_montage.png)", "", "![地面纹理容量对照](capacity_paving_montage.png)", "",
            "[全部 10 个区域容量合图](capacity_all_regions.png)", "", "## 单独区域放大图", ""]
    for region in regions:
        frame = region["frame"]
        index = 0 if region["label"].startswith("A ") else 1
        text.append(f"- [帧 {frame:04d} / {region['label']} / 2 倍放大](regions_zoom2x/frame{frame:04d}_roi{index}.png)")
    text += ["", "生成命令：", "", "```powershell", "& 'D:/anaconda3/envs/qsr-gaming/python.exe' -u scripts/16_make_medium_roi_gallery.py --device cuda", "```"]
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Multi-ROI gallery complete: {out}", flush=True)


if __name__ == "__main__":
    main()
