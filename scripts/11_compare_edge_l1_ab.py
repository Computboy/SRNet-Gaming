"""After A/B finish, verify full validation results, plot curves and fixed ROIs.

Uses only train logs and Scene_1_02 validation. Selection rules are never changed
here and Test is not read. Metrics are verified from each selected epoch's 240-row
validation CSV. SSIM is computed after selection on all 240 frames for each
distinct selected checkpoint; five predeclared frames are also visualized.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset
from src.edge_validation import json_safe, select_epochs, EdgeValidationProtocol
from src.trainer import build_quicksrnet_from_local_checkpoint
from src.losses import TrainingLossProtocol
from src.post_selection import selected_ssim


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_comparable_configs(a, b, *, a_weight=0.0, b_weight=0.1):
    for key in ("model", "data", "validation", "seed"):
        if a.get(key) != b.get(key):
            raise ValueError(f"A/B mismatch in {key}")
    ta, tb = dict(a["training"]), dict(b["training"])
    for settings in (ta, tb):
        settings.pop("epochs", None)  # load_run checks the completed epoch budget.
        settings.pop("loss", None)
        settings.pop("edge_l1_weight", None)
    if ta != tb:
        raise ValueError("A/B training hyperparameters differ beyond the loss")
    for label, config, weight in (("A", a, a_weight), ("B", b, b_weight)):
        expected = TrainingLossProtocol("l1" if weight == 0 else "l1_edge", weight)
        if TrainingLossProtocol.from_config(config) != expected:
            raise ValueError(f"{label} must use the declared loss {expected.mode} with edge_l1_weight={weight:g}")


def load_run(run, epochs):
    checkpoint = torch.load(ROOT / "checkpoints/finetuned" / run / "last.pth", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != epochs or not checkpoint["validation_is_independent"]:
        raise ValueError(f"Not a completed {epochs}-epoch formal run: {run}")
    history = read_csv(ROOT / "outputs/metrics" / run / "metrics.csv")
    if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
        raise ValueError(f"Incomplete epoch CSV: {run}")
    summary = json.loads((ROOT / "outputs/metrics" / run / "selection_summary.json").read_text(encoding="utf-8"))
    choices = select_epochs(checkpoint["metrics"], EdgeValidationProtocol.from_config(checkpoint["config"]))
    for strategy, choice in choices.items():
        if summary["selected"][strategy]["epoch"] != choice["epoch"]:
            raise ValueError(f"Selection mismatch: {run}/{strategy}")
    manifest = json.loads((ROOT / "outputs/logs" / run / "validation_manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != checkpoint["validation_fingerprint"]:
        raise ValueError(f"Validation manifest mismatch: {run}")
    return {"run": run, "checkpoint": checkpoint, "history": history, "summary": summary, "manifest": manifest}


def selected_validation(data, label):
    expected_keys = {item["key"] for item in data["manifest"]["pairs"]}
    if len(expected_keys) != 240:
        raise ValueError("Expected all 240 validation frames")
    records = []
    per_frame = {}
    for strategy, choice in data["summary"]["selected"].items():
        epoch = int(choice["epoch"])
        rows = read_csv(ROOT / "outputs/metrics" / data["run"] / "validation" / f"epoch{epoch:03d}.csv")
        if len(rows) != 240 or {row["key"] for row in rows} != expected_keys:
            raise ValueError(f"Incomplete selected validation data: {data['run']}/{epoch}")
        posthoc = data.get("post_selection_ssim", {}).get(epoch)
        if posthoc:
            rows = posthoc["rows"]
            if len(rows) != 240 or {row["key"] for row in rows} != expected_keys:
                raise ValueError("Incomplete post-selection validation data")
        if sum(row.get("ssim") not in ("", None) for row in rows) != 240:
            raise ValueError("Missing complete post-selection SSIM evaluation")
        mapping = {row["key"]: row for row in rows}
        means = {}
        for column, summary_key in (("psnr_db", "val_psnr_db"), ("ssim", "val_ssim"),
                                    ("edge_region_psnr_db", "val_edge_region_psnr_db")):
            values = [float(row[column]) for row in rows if row[column] not in ("", None)]
            mean = sum(values) / len(values) if values else None
            saved_mean = choice.get(summary_key)
            if mean is not None and saved_mean is not None and not math.isclose(mean, float(saved_mean), abs_tol=1e-5, rel_tol=1e-7):
                raise ValueError(f"Per-frame mean differs from saved selection: {data['run']}/{strategy}/{column}")
            means[summary_key] = mean
        model = torch.load(choice["checkpoint"], map_location="cpu", weights_only=True)
        if model["epoch"] != epoch:
            raise ValueError("Selected checkpoint epoch differs from its metadata")
        records.append({"experiment": label, "run": data["run"], "strategy": strategy, "epoch": epoch,
                        **means, "ssim_source": "post_selection" if posthoc else "per_epoch",
                        "val_weighted_score": choice["val_weighted_score"], "checkpoint": choice["checkpoint"]})
        per_frame[strategy] = mapping
    return records, per_frame


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loss_label(weight):
    return "L1" if weight == 0 else f"L1 + {weight:g} × Edge-L1"


def plot_curves(a, b, out, *, a_weight=0.0, b_weight=0.1):
    series = ((f"A: {loss_label(a_weight)}", a, "#2166ac"),
              (f"B: {loss_label(b_weight)}", b, "#c45a1a"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fields = [("val_psnr_db", "Full-image PSNR (dB)"), ("val_edge_region_psnr_db", "Edge-region PSNR (dB)"),
              ("val_loss", "Validation RGB L1"), ("val_weighted_score", "Weighted validation score")]
    for ax, (key, label) in zip(axes.flat, fields):
        for name, data, color in series:
            x = [int(row["epoch"]) for row in data["history"]]
            y = [float(row[key]) if row.get(key) not in ("", None) else np.nan for row in data["history"]]
            ax.plot(x, y, label=name, color=color, linewidth=1.6)
            for strategy, marker in (("psnr", "o"), ("weighted", "s"), ("edge_constrained", "^")):
                epoch = data["summary"]["selected"][strategy]["epoch"]
                ax.scatter(epoch, y[epoch - 1], color=color, marker=marker, s=65, facecolors="none", linewidths=1.4)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("QuickSRNet A/B | identical validation protocol | circle: PSNR, square: weighted, triangle: constrained")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"validation_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, (("train_global_l1", "Global L1"), ("train_edge_l1", "Unweighted edge L1"),
                                      ("train_total_loss", "Training objective (different definitions)"))):
        for name, data, color in series:
            values = [float(row[key]) if row.get(key) else np.nan for row in data["history"]]
            ax.plot(range(1, len(values) + 1), values, label=name, color=color)
        ax.set(xlabel="Epoch", ylabel="Mean loss", title=title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"loss_decomposition.{suffix}", dpi=180)
    plt.close(fig)


def plot_selected_ssim(records, out):
    strategies = ("psnr", "weighted", "edge_constrained")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(strategies))
    for label, offset, color in (("A", -0.18, "#2166ac"), ("B", 0.18, "#c45a1a")):
        values = [next(row["val_ssim"] for row in records
                       if row["experiment"] == label and row["strategy"] == strategy) for strategy in strategies]
        ax.bar(x + offset, values, width=0.36, label=label, color=color)
    ax.set(xticks=x, xticklabels=strategies, ylabel="Full-validation RGB SSIM",
           title="SSIM evaluated after checkpoint selection (240 frames/model)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"selected_ssim_comparison.{suffix}", dpi=180)
    plt.close(fig)


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def image(tensor):
    array = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def visualize(a, b, records, diagnostic_cases, out):
    plan = json.loads((ROOT / "outputs/verification/val-scene1-02-20261019-140251/sample_plan.json").read_text(encoding="utf-8"))
    if plan["frames"] != [0, 60, 120, 180, 239]:
        raise ValueError("Fixed validation visualization plan changed")
    settings = a["checkpoint"]["config"]
    model_name = settings["model"]["name"]
    medium = model_name == "quicksrnet_medium"
    roi_records = [record for record in records if record["strategy"] == "psnr"] if medium else records
    dataset = PairedImageDataset(ROOT / settings["data"]["val_lr"], ROOT / settings["data"]["val_hr"])
    if len(dataset) != 240 or any(not pair.key.startswith("scene_1_02/") for pair in dataset.pairs):
        raise ValueError("Visualization is restricted to Scene_1_02 validation")
    sample_dir = out / "samples"
    sample_dir.mkdir()
    pretrained = ROOT / settings["model"]["pretrained_checkpoint"]
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for record in records:
        key = (record["experiment"], record["epoch"])
        if key not in models:
            checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
            model = build_quicksrnet_from_local_checkpoint(pretrained, model_name=model_name)
            model.model.load_state_dict(checkpoint["state_dict"], strict=True)
            models[key] = model.to(device).float().eval().requires_grad_(False)
    with torch.inference_mode():
        for frame in plan["frames"]:
            item = dataset[frame]
            lr = item["lr"].unsqueeze(0).to(device)
            hr = item["hr"].unsqueeze(0).to(device)
            outputs = {key: model(lr).clamp(0, 1) for key, model in models.items()}
            reference = image(hr)
            lr_display = image(lr).resize(reference.size, Image.Resampling.NEAREST) if medium else None
            for roi_index, (roi_name, box) in enumerate(plan["rois"][str(frame)]):
                reference_columns = 2 if medium else 1
                canvas = Image.new("RGB", (280 * (reference_columns + len(roi_records)), 350), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 5), f"Validation frame {frame:04d} | {roi_name} | Full-frame scores (dB)", fill="#17212b", font=font(17))
                hr_left = 288 if medium else 8
                draw.text((hr_left, 35), "HR reference", fill="#17212b", font=font(16))
                canvas.paste(reference.crop(box), (hr_left, 86))
                if lr_display is not None:
                    draw.text((8, 35), "LR input (nearest 2x display)", fill="#17212b", font=font(16))
                    canvas.paste(lr_display.crop(box), (8, 86))
                for column, record in enumerate(roi_records, start=reference_columns):
                    x = column * 280
                    label = f"{record['experiment']} {record['strategy']}"
                    if medium:
                        group = a if record["experiment"] == "A" else b
                        weight = TrainingLossProtocol.from_config(group["checkpoint"]["config"]).edge_l1_weight
                        label = f"Medium {'L1' if weight == 0 else f'Edge-L1 (λ={weight:g})'}"
                    draw.text((x + 8, 35), f"{label} | E{record['epoch']}", fill="#17212b", font=font(15))
                    frame_rows = read_csv(ROOT / "outputs/metrics" / record["run"] / "validation" / f"epoch{record['epoch']:03d}.csv")
                    score = next(row for row in frame_rows if row["key"] == item["name"])
                    draw.text((x + 8, 59), f"P {float(score['psnr_db']):.3f} | Edge {float(score['edge_region_psnr_db']):.3f}", fill="#44515f", font=font(15))
                    canvas.paste(image(outputs[(record["experiment"], record["epoch"])]).crop(box), (x + 8, 86))
                canvas.save(sample_dir / f"frame{frame:04d}_roi_{roi_index}.png")
        diagnostic_dir = sample_dir / "metric_selected"
        diagnostic_dir.mkdir()
        for case in diagnostic_cases:
            frame = int(Path(case["key"]).stem.replace("frame_", ""))
            item = dataset[frame]
            if item["name"] != case["key"]:
                raise ValueError("Diagnostic frame index differs from its fixed key")
            lr = item["lr"].unsqueeze(0).to(device)
            panels = [("HR reference", item["hr"].unsqueeze(0))]
            for label in ("A", "B"):
                record = next(row for row in records if row["experiment"] == label and row["strategy"] == case["strategy"])
                panels.append((f"{label} {case['strategy']} | epoch {record['epoch']}", models[(label, record["epoch"])](lr)))
            canvas = Image.new("RGB", (960 * 3, 605), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 5), f"METRIC-SELECTED diagnostic | {case['reason']} | frame {frame:04d} | not a representative sample", fill="#17212b", font=font(22))
            for column, (label, tensor) in enumerate(panels):
                draw.text((column * 960 + 10, 35), label, fill="#17212b", font=font(20))
                canvas.paste(image(tensor).resize((960, 540), Image.Resampling.BICUBIC), (column * 960, 65))
            canvas.save(diagnostic_dir / f"{case['strategy']}_{case['reason']}_frame{frame:04d}.png")
    (out / "sample_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--b-run", required=True)
    parser.add_argument("--a-edge-l1-weight", type=float, default=0.0)
    parser.add_argument("--b-edge-l1-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError("Choose a new comparison output directory")
    a, b = load_run(args.a_run, args.epochs), load_run(args.b_run, args.epochs)
    if a["checkpoint"]["validation_fingerprint"] != b["checkpoint"]["validation_fingerprint"]:
        raise ValueError("A/B used different validation contents")
    if a["summary"]["protocol"] != b["summary"]["protocol"]:
        raise ValueError("A/B used different validation definitions")
    verify_comparable_configs(a["checkpoint"]["config"], b["checkpoint"]["config"],
                              a_weight=args.a_edge_l1_weight, b_weight=args.b_edge_l1_weight)
    hashes = [data["checkpoint"]["config"]["runtime"]["pretrained_checkpoint_sha256"] for data in (a, b)]
    if hashes[0] != hashes[1]:
        raise ValueError("A/B initialization checksums differ")
    out.mkdir(parents=True)
    for label, data in (("A", a), ("B", b)):
        model_name = data["checkpoint"]["config"]["model"]["name"]
        model_factory = partial(build_quicksrnet_from_local_checkpoint, model_name=model_name)
        selected_ssim(data, ROOT, out / "post_selection_ssim" / label, read_csv, save_csv,
                      model_factory)
    records_a, frames_a = selected_validation(a, "A")
    records_b, frames_b = selected_validation(b, "B")
    records = records_a + records_b
    deltas = []
    for strategy in frames_a:
        ra = next(row for row in records_a if row["strategy"] == strategy)
        rb = next(row for row in records_b if row["strategy"] == strategy)
        deltas.append({"strategy": strategy, "A_epoch": ra["epoch"], "B_epoch": rb["epoch"],
            **{f"B_minus_A_{key}": rb[key] - ra[key] for key in ("val_psnr_db", "val_ssim", "val_edge_region_psnr_db")}})
    save_csv(out / "selected_models.csv", records)
    save_csv(out / "paired_strategy_deltas.csv", deltas)
    paired = []
    for strategy in frames_a:
        for key, ar in frames_a[strategy].items():
            br = frames_b[strategy][key]
            paired.append({"strategy": strategy, "key": key,
                **{f"{label}_{metric}": float(row[metric]) if row[metric] not in ("", None) else None
                   for label, row in (("A", ar), ("B", br)) for metric in ("psnr_db", "ssim", "edge_region_psnr_db")}})
    diagnostic_cases, regression_counts = [], {}
    for strategy in frames_a:
        items = [row for row in paired if row["strategy"] == strategy]
        for item in items:
            for metric in ("psnr_db", "ssim", "edge_region_psnr_db"):
                item[f"B_minus_A_{metric}"] = item[f"B_{metric}"] - item[f"A_{metric}"]
        regression_counts[strategy] = {metric: sum(row[f"B_minus_A_{metric}"] < 0 for row in items)
                                      for metric in ("psnr_db", "ssim", "edge_region_psnr_db")}
        for reason, metric in (("minimum_psnr_delta", "psnr_db"), ("minimum_ssim_delta", "ssim"),
                               ("minimum_edge_psnr_delta", "edge_region_psnr_db")):
            diagnostic_cases.append({**min(items, key=lambda row: row[f"B_minus_A_{metric}"]), "reason": reason})
        tradeoff = [row for row in items if row["B_minus_A_edge_region_psnr_db"] > 0 and row["B_minus_A_ssim"] < 0]
        if tradeoff:
            diagnostic_cases.append({**max(tradeoff, key=lambda row: row["B_minus_A_edge_region_psnr_db"]),
                                     "reason": "edge_gain_with_ssim_drop"})
    save_csv(out / "paired_validation_frames.csv", paired)
    save_csv(out / "metric_selected_cases.csv", diagnostic_cases)
    plot_curves(a, b, out, a_weight=args.a_edge_l1_weight, b_weight=args.b_edge_l1_weight)
    plot_selected_ssim(records, out)
    for label, data in (("A", a), ("B", b)):
        subprocess.run([sys.executable, str(ROOT / "scripts/09_plot_selection_comparison.py"), "--metrics",
            str(ROOT / "outputs/metrics" / data["run"] / "metrics.csv"), "--output-dir", str(out / label),
            "--selected-ssim", str(out / "selected_models.csv")], check=True)
    visualize(a, b, records, diagnostic_cases, out)
    summary = {"epochs_per_run": args.epochs, "validation_frames": 240, "selected": records, "paired_deltas": deltas,
        "edge_l1_weights": {"A": args.a_edge_l1_weight, "B": args.b_edge_l1_weight},
        "same_official_initialization_sha256": hashes[0], "same_validation_fingerprint": a["checkpoint"]["validation_fingerprint"],
        "validation_protocol": a["summary"]["protocol"], "test_accessed": False,
        "ssim_schedule": "post_selection; full 240-frame evaluation for each distinct selected checkpoint",
        "per_frame_regression_counts": regression_counts, "metric_selected_cases": diagnostic_cases,
        "resume_epochs": {label: [int(row["resumed_from_epoch"]) for row in data["history"] if int(row.get("resumed_from_epoch") or 0)]
                          for label, data in (("A", a), ("B", b))},
        "limitation": "One seed, one game environment; validation selected checkpoints; no aliasing-only or temporal metric"}
    (out / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = [f"# QuickSRNet A/B 验证比较 · {loss_label(args.a_edge_l1_weight)} 与 {loss_label(args.b_edge_l1_weight)}", "",
        f"A 组为 `{args.a_run}`（λ={args.a_edge_l1_weight:g}），B 组为 `{args.b_run}`（λ={args.b_edge_l1_weight:g}）。", "",
        f"两组各 {args.epochs} Epoch，使用同一官方权重、数据划分、随机种子与验证协议。下表每项均为完整 240 帧验证结果。", "",
        "| 组别 | 选模规则 | Epoch | PSNR (dB) | SSIM | Edge-region PSNR (dB) |", "|---|---|---:|---:|---:|---:|"]
    report += [f"| {r['experiment']} | {r['strategy']} | {r['epoch']} | {r['val_psnr_db']:.6f} | {r['val_ssim']:.6f} | {r['val_edge_region_psnr_db']:.6f} |" for r in records]
    report += ["", "## 同一选模规则下 B−A", "", "| 规则 | ΔPSNR (dB) | ΔSSIM | ΔEdge-region PSNR (dB) |", "|---|---:|---:|---:|"]
    report += [f"| {r['strategy']} | {r['B_minus_A_val_psnr_db']:+.6f} | {r['B_minus_A_val_ssim']:+.6f} | {r['B_minus_A_val_edge_region_psnr_db']:+.6f} |" for r in deltas]
    report += ["", "## 可能退步与指标取舍的诊断", "",
               "下列数量是逐帧指标差值为负的帧数；连续帧相关，不能作为独立样本显著性。", "",
               "| 规则 | PSNR 更低的帧数 | SSIM 更低的帧数 | 边缘 PSNR 更低的帧数 |", "|---|---:|---:|---:|"]
    report += [f"| {strategy} | {counts['psnr_db']} | {counts['ssim']} | {counts['edge_region_psnr_db']} |"
               for strategy, counts in regression_counts.items()]
    report += ["", "附加导出整帧 PSNR、SSIM 和边缘 PSNR 增量各自最小的诊断帧，以及边缘分数提高且 SSIM 下降的代表帧。",
               "它们明确标记为按指标选择，不能冒充代表性样本；若最小增量仍为正，也不能称其为退步帧。",
               "诊断图在 samples/metric_selected/；各帧差值见 metric_selected_cases.csv。这些是待目视复核的候选，不是自动判断出的视觉优劣。"]
    report += ["", "## 解读边界", "",
        "SSIM 在选模后对每个不同的入选 checkpoint 完整评估 240 帧，不参与选模。结果另存于 post_selection_ssim/，未回写训练历史或选模权重。前期历史 SSIM 记录保留；后续空白表示后置评估，不是 0，也不插值生成逐轮 SSIM 曲线。",
        "这些是用于选模的验证结果，尚不构成独立 Test 结论。不能从不同选模规则任意择优再声称控制变量。",
        "固定 ROI 复用事先声明的五帧及相同 HR 坐标；图上分数为对应整帧指标。需查看轮廓、植被、纹理、细线、粗台阶、亮边与模糊，记录可能恶化的例子。",
        "若边缘分数上升但 SSIM 下降或画面更平滑，应表述为边缘区域像素误差改善，不得据此声称抗锯齿已改善。",
        "同一随机种子与采样配置并不保证 GPU 算法逐位确定；单种子不能说明训练随机性范围。连续 240 帧也不是 240 个独立场景。",
        "如发生恢复训练，checkpoint 会恢复模型、优化器、AMP 和历史指标；原管线未保存所有 RNG/DataLoader worker 随机状态，因此不保证与不中断训练逐位一致。实际恢复位置另记于 summary.json。",
        "本脚本未访问 Test，也未搜索 λ、α、阈值、mask 宽度或 PSNR 容忍区间。"]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"A/B validation analysis completed: {out}")


if __name__ == "__main__":
    main()
