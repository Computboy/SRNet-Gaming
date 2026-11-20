"""After C completes: reuse validation reporting and compare Small/Medium L1."""
from __future__ import annotations
import argparse
import functools
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from src.capacity import benchmark_pair, model_description, sha256, verify_capacity_configs
from src.dataset import PairedImageDataset
from src.edge_validation import json_safe
from src.post_selection import selected_ssim
from src.trainer import build_quicksrnet_from_local_checkpoint


def reporting_helpers():
    spec = importlib.util.spec_from_file_location("existing_validation_reporting", ROOT / "scripts/11_compare_edge_l1_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_capacity(a, c, records, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title) in zip(axes.flat, (("train_global_l1", "Training RGB L1"),
            ("val_loss", "Validation RGB L1"), ("val_psnr_db", "Validation PSNR (dB)"),
            ("val_edge_region_psnr_db", "Validation edge-region PSNR (dB)"))):
        for label, data, color in (("A: Small + L1", a, "#2166ac"), ("C: Medium + L1", c, "#c45a1a")):
            ax.plot([int(r["epoch"]) for r in data["history"]], [float(r[field]) for r in data["history"]], label=label, color=color)
        ax.set(xlabel="Epoch", ylabel=title)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("QuickSRNet capacity ablation | same fine-tuning and validation settings")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"capacity_training_curves.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    strategies = ("psnr", "weighted", "edge_constrained")
    for label, offset, color in (("A", -.18, "#2166ac"), ("C", .18, "#c45a1a")):
        values = [next(r["val_ssim"] for r in records if r["experiment"] == label and r["strategy"] == strategy) for strategy in strategies]
        ax.bar(np.arange(3) + offset, values, width=.36, label=label, color=color)
    ax.set(xticks=np.arange(3), xticklabels=strategies, ylabel="Full-validation RGB SSIM", title="SSIM evaluated after selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"selected_ssim_comparison.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="asian-village-selection-001")
    parser.add_argument("--candidate-run", default="asian-village-medium-l1-001")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Choose a new capacity report output directory; existing results are never overwritten")
    report = reporting_helpers()
    a, c = report.load_run(args.baseline_run, 100), report.load_run(args.candidate_run, 100)
    fairness = verify_capacity_configs(a["checkpoint"]["config"], c["checkpoint"]["config"])
    if a["checkpoint"]["validation_fingerprint"] != c["checkpoint"]["validation_fingerprint"]:
        raise ValueError("A/C used different validation contents")
    for data in (a, c):
        config = data["checkpoint"]["config"]
        if sha256(ROOT / config["model"]["pretrained_checkpoint"]) != config["runtime"]["pretrained_checkpoint_sha256"]:
            raise ValueError("Official initialization differs from the saved training record")
    output.mkdir(parents=True)
    records, frames = [], {}
    for label, data in (("A", a), ("C", c)):
        factory = functools.partial(build_quicksrnet_from_local_checkpoint, model_name=data["checkpoint"]["config"]["model"]["name"])
        selected_ssim(data, ROOT, output / "post_selection_ssim" / label, report.read_csv, report.save_csv, factory)
        chosen, mapping = report.selected_validation(data, label)
        records.extend(chosen)
        frames[label] = mapping
    report.save_csv(output / "selected_models.csv", records)
    deltas, paired = [], []
    strategies = ("psnr", "weighted", "edge_constrained")
    for strategy in strategies:
        ar = next(r for r in records if r["experiment"] == "A" and r["strategy"] == strategy)
        cr = next(r for r in records if r["experiment"] == "C" and r["strategy"] == strategy)
        deltas.append({"strategy": strategy, "A_epoch": ar["epoch"], "C_epoch": cr["epoch"],
                       **{f"C_minus_A_{metric}": cr[metric] - ar[metric] for metric in ("val_psnr_db", "val_ssim", "val_edge_region_psnr_db")}})
        for key, row in frames["A"][strategy].items():
            other = frames["C"][strategy][key]
            paired.append({"strategy": strategy, "key": key,
                           **{f"{label}_{metric}": float(source[metric]) if source[metric] not in (None, "") else None
                              for label, source in (("A", row), ("C", other)) for metric in ("psnr_db", "ssim", "edge_region_psnr_db")}})
    report.save_csv(output / "paired_strategy_deltas.csv", deltas)
    report.save_csv(output / "paired_validation_frames.csv", paired)
    plot_capacity(a, c, records, output)
    for label, data in (("A", a), ("C", c)):
        subprocess.run([sys.executable, str(ROOT / "scripts/09_plot_selection_comparison.py"), "--metrics",
                        str(ROOT / "outputs/metrics" / data["run"] / "metrics.csv"), "--output-dir", str(output / label),
                        "--selected-ssim", str(output / "selected_models.csv")], check=True)
    settings = c["checkpoint"]["config"]
    dataset = PairedImageDataset(ROOT / settings["data"]["val_lr"], ROOT / settings["data"]["val_hr"])
    plan = json.loads((ROOT / "outputs/verification/val-scene1-02-20261019-140251/sample_plan.json").read_text(encoding="utf-8"))
    if plan["frames"] != [0, 60, 120, 180, 239]:
        raise ValueError("Fixed ROI plan changed")
    device = torch.device("cuda")
    models = {}
    for label, data in (("A", a), ("C", c)):
        config = data["checkpoint"]["config"]
        for record in (r for r in records if r["experiment"] == label):
            key = (label, record["epoch"])
            if key not in models:
                model = build_quicksrnet_from_local_checkpoint(ROOT / config["model"]["pretrained_checkpoint"], model_name=config["model"]["name"])
                saved = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
                model.model.load_state_dict(saved["state_dict"], strict=True)
                models[key] = model.to(device).float().eval()
    sample = dataset[0]["lr"].unsqueeze(0).to(device)
    timings, timed_pairs, descriptions = {}, {}, {}
    for strategy in strategies:
        selected = {label: next(r for r in records if r["experiment"] == label and r["strategy"] == strategy) for label in ("A", "C")}
        pair = (selected["A"]["epoch"], selected["C"]["epoch"])
        if pair not in timed_pairs:
            timed_pairs[pair] = benchmark_pair({label: models[(label, record["epoch"])] for label, record in selected.items()}, sample,
                                               warmup=20, iterations=50, rounds=5)
        timings[strategy] = {"A_epoch": pair[0], "C_epoch": pair[1], **timed_pairs[pair]}
        for label, record in selected.items():
            descriptions[label] = model_description(models[(label, record["epoch"])])
    (output / "inference_latency.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    sample_dir = output / "samples"
    sample_dir.mkdir()
    with torch.inference_mode():
        for index in plan["frames"]:
            item = dataset[index]
            lr = item["lr"].unsqueeze(0).to(device)
            predictions = {key: report.image(model(lr)) for key, model in models.items()}
            hr = report.image(item["hr"].unsqueeze(0))
            for strategy in strategies:
                chosen = [next(r for r in records if r["experiment"] == label and r["strategy"] == strategy) for label in ("A", "C")]
                for roi_index, (roi_name, box) in enumerate(plan["rois"][str(index)]):
                    canvas = Image.new("RGB", (840, 345), "white")
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 5), f"Frame {index:04d} | {roi_name} | {strategy} | identical HR coordinates", font=report.font(14), fill="#17212b")
                    draw.text((8, 35), "HR reference", font=report.font(16), fill="#17212b")
                    canvas.paste(hr.crop(box), (8, 78))
                    for column, record in enumerate(chosen, 1):
                        label = record["experiment"]
                        draw.text((column * 280 + 8, 35), f"{label}: {'Small' if label == 'A' else 'Medium'} | E{record['epoch']}", font=report.font(16), fill="#17212b")
                        canvas.paste(predictions[(label, record["epoch"])].crop(box), (column * 280 + 8, 78))
                    canvas.save(sample_dir / f"{strategy}_frame{index:04d}_roi{roi_index}.png")
    summary = {"selected": records, "paired_deltas": deltas, "model_descriptions": descriptions,
               "fairness": fairness, "inference_timing": "inference_latency.json", "test_accessed": False,
               "resume_epochs": {label: [int(r["resumed_from_epoch"]) for r in data["history"] if int(r.get("resumed_from_epoch") or 0)] for label, data in (("A", a), ("C", c))}}
    (output / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# A/C 模型容量消融：Small + L1 与 Medium + L1", "", "全部质量指标来自相同的 240 张完整验证图，SSIM 在选模后计算。", "",
             "| 组别 | 规则 | Epoch | PSNR | SSIM | Edge-region PSNR |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['experiment']} | {r['strategy']} | {r['epoch']} | {r['val_psnr_db']:.6f} | {r['val_ssim']:.6f} | {r['val_edge_region_psnr_db']:.6f} |" for r in records]
    lines += ["", "## 参数量与实际推理耗时", "", "输入为同一验证帧 LR 960×540，batch=1；同一 GPU，FP32、AMP/TF32 关闭。CUDA event 只计前向，不计解码、传输、指标或存盘。预热 20 次，5 轮交替顺序，每轮 50 次。", "",
              "| 规则 | A 参数量 | C 参数量 | A 中位耗时 ms | C 中位耗时 ms |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {strategy} | {descriptions['A']['parameters']} | {descriptions['C']['parameters']} | {timings[strategy]['models']['A']['median_ms']:.4f} | {timings[strategy]['models']['C']['median_ms']:.4f} |" for strategy in strategies]
    lines += ["", "GPU、驱动、温度、时钟、完整逐次耗时及分位数见 inference_latency.json。质量分数使用原 FP32 验证器；该计时只用于同条件性能比较。", "",
              "容量与各自官方预训练权重同时不同，不能将质量差异完全归因于参数量。A 在第 11 轮后断点续训且未保留完整 RNG/worker 状态；实际恢复轮次见 summary.json。单种子、单场景不代表统计显著性。", "",
              "mask 定义、阈值、膨胀、加权系数与容忍区间沿用 A；基线代码没有 mask 缓存，本实验未新增。未使用 Test 调参、选模或报告独立测试结论。", "",
              "曲线见 capacity_training_curves.png；SSIM 见 selected_ssim_comparison.png；samples/ 为固定五帧、相同 ROI 的 HR/A/C 三列图。"]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Capacity analysis complete: {output}", flush=True)


if __name__ == "__main__":
    main()
