"""Plot the stage-1 validation history and compare checkpoint selection rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-ssim", type=Path, help="Post-selection selected_models.csv for report-only SSIM")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Choose a new plot output directory")
    with args.metrics.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "val_edge_region_psnr_db" not in rows[0]:
        raise ValueError("Use metrics.csv from a run with edge_evaluation enabled")
    summary = json.loads((args.metrics.parent / "selection_summary.json").read_text(encoding="utf-8"))
    reported_ssim = {}
    if args.selected_ssim:
        with args.selected_ssim.open(newline="", encoding="utf-8-sig") as handle:
            reported_ssim = {row["strategy"]: float(row["val_ssim"]) for row in csv.DictReader(handle)
                             if row["run"] == args.metrics.parent.name}
    epochs = np.array([int(row["epoch"]) for row in rows])
    if epochs.tolist() != list(range(1, int(epochs[-1]) + 1)):
        raise ValueError("Expected consecutive complete epoch history")
    if int(summary["completed_epoch"]) != int(epochs[-1]):
        raise ValueError("Selection summary and CSV have different completed epochs")
    fields = [("val_psnr_db", "Full-image RGB PSNR (dB)"),
              ("val_edge_region_psnr_db", "Edge-region RGB PSNR (dB)"),
              ("val_ssim", "RGB SSIM"), ("val_weighted_score", "Weighted validation score")]
    series = {}
    for key, _ in fields:
        values = np.array([float(row[key]) if row.get(key) else np.nan for row in rows])
        series[key] = np.where(np.isfinite(values), values, np.nan)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    styles = {"psnr": ("#1f77b4", "o"), "weighted": ("#d97706", "s"),
              "edge_constrained": ("#7c3aed", "^")}
    for ax, (key, label) in zip(axes.flat, fields):
        ax.plot(epochs, series[key], color="#167d73", linewidth=1.5, marker=".")
        if key == "val_ssim" and any(not row.get(key) for row in rows):
            ax.text(0.03, 0.03, "Missing epochs: SSIM deferred until selection", transform=ax.transAxes, fontsize=8)
        for name, choice in summary["selected"].items():
            index = int(choice["epoch"]) - 1
            color, marker = styles[name]
            ax.scatter([epochs[index]], [series[key][index]], color=color, marker=marker, s=85,
                       facecolors="none", linewidths=1.5, zorder=5,
                       label=f"{name}: epoch {epochs[index]}")
        for row in rows:
            if int(row.get("resumed_from_epoch") or 0):
                ax.axvline(int(row["epoch"]) - 0.5, color="#64748b", linestyle="--", linewidth=1)
        ax.set(xlabel="Epoch", ylabel=label, xlim=(0.5, max(1.5, epochs[-1] + 0.5)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    scope = "Independent validation" if summary["validation_is_independent"] else "Training-pair SELF-CHECK"
    protocol = summary["protocol"]
    fig.suptitle(f"QuickSRNet selection comparison | {scope}\n"
                 f"alpha={protocol['weighted_alpha']}, PSNR tolerance={protocol['psnr_tolerance_db']} dB")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output_dir.mkdir(parents=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"selection_curves.{suffix}", dpi=180)
    plt.close(fig)
    comparison = [{"strategy": name, **{key: choice.get(key) for key in (
        "epoch", "val_psnr_db", "val_ssim", "val_edge_region_psnr_db", "val_weighted_score", "checkpoint")}}
        for name, choice in summary["selected"].items()]
    for row in comparison:
        if row["strategy"] in reported_ssim:
            row["val_ssim"] = reported_ssim[row["strategy"]]
    with (args.output_dir / "selected_models.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    report = ["# 验证选模策略比较", "", f"评估范围：{scope}。", "",
        "| 策略 | Epoch | PSNR (dB) | Edge-region PSNR (dB) | SSIM |",
        "|---|---:|---:|---:|---:|"]
    for row in comparison:
        edge = row["val_edge_region_psnr_db"]
        report.append(f"| {row['strategy']} | {row['epoch']} | {float(row['val_psnr_db']):.6f} | {edge} | {row['val_ssim']} |")
    report.extend(["", f"三种策略是否选择同一 Epoch：{summary['same_epoch_all_strategies']}。", "",
        "这是同一条训练轨迹上的验证选择比较，不是独立测试结论。",
        "SSIM 不参与选模；空白为后置评估，历史曲线只显示实际计算过的轮次。传入的后置 SSIM 仅用于入选模型表格。",
        "alpha、PSNR 容忍阈值与边缘定义均为实验设置，不能据本图声称其最优。"])
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved selection curves and table: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
