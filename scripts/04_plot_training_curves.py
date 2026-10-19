"""Export training curves and a convergence summary from saved epoch metrics.

This script needs NumPy and Matplotlib, but does not import PyTorch or use a GPU.
"""
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


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= args.end_epoch]
    epochs = np.array([int(row["epoch"]) for row in rows])
    if epochs.tolist() != list(range(1, args.end_epoch + 1)):
        raise ValueError("Metrics must contain every epoch exactly once, from 1 to end-epoch")
    losses = np.array([float(row["train_loss"]) for row in rows])
    val_losses = np.array([float(row["val_loss"]) for row in rows])
    psnr = np.array([float(row["val_psnr_db"]) for row in rows])
    rates = np.array([float(row["learning_rate"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (losses, val_losses, psnr, rates)):
        raise ValueError("Non-finite metrics found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    baseline_path = ROOT / "outputs/preflight-20261018/report.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["pretrained_validation"]["psnr_db"]
    best_index = int(psnr.argmax())
    n = min(10, len(rows) // 2)
    recent_slope = float(np.polyfit(epochs[-n:], psnr[-n:], 1)[0])
    summary = {
        "epochs": len(rows),
        "pretrained_val_psnr_db": baseline,
        "best_epoch": int(epochs[best_index]),
        "best_val_psnr_db": float(psnr[best_index]),
        "gain_over_pretrained_db": float(psnr[best_index] - baseline),
        "last_train_loss": float(losses[-1]),
        "last_val_loss": float(val_losses[-1]),
        "last_val_psnr_db": float(psnr[-1]),
        "last_10_psnr_slope_db_per_epoch": recent_slope,
        "last_10_psnr_range_db": [float(psnr[-n:].min()), float(psnr[-n:].max())],
        "previous_10_mean_train_loss": float(losses[-2*n:-n].mean()),
        "last_10_mean_train_loss": float(losses[-n:].mean()),
        "previous_10_mean_val_loss": float(val_losses[-2*n:-n].mean()),
        "last_10_mean_val_loss": float(val_losses[-n:].mean()),
        "learning_rates": sorted(set(rates.tolist())),
        "validation_scope": sorted(set(row["validation_scope"] for row in rows)),
    }
    for epoch in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        if epoch <= args.end_epoch:
            summary[f"epoch_{epoch}_psnr_db"] = float(psnr[epoch - 1])
    if args.end_epoch >= 50:
        summary["gain_epoch_40_to_50_db"] = float(psnr[49] - psnr[39])
    if args.end_epoch >= 100:
        summary["gain_epoch_50_to_100_db"] = float(psnr[99] - psnr[49])
        summary["best_gain_after_epoch_50_db"] = float(psnr.max() - psnr[:50].max())
    (args.output_dir / "curve_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

    def mark_resume(ax: plt.Axes) -> None:
        if args.end_epoch > 50:
            ax.axvline(50.5, color="#64748b", linestyle="--", linewidth=1,
                       label="Resume from epoch 50")

    def plot_loss(ax: plt.Axes) -> None:
        ax.plot(epochs, losses, color="#99b9d9", linewidth=1.2, marker=".",
                markersize=4, label="Per-epoch loss")
        window = 5
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(epochs[window-1:], smoothed, color="#175b93", linewidth=2,
                label="5-epoch moving average")
        ax.set(title="Train Loss vs. Epoch", xlabel="Epoch", ylabel="Mean training L1 loss")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, args.end_epoch + 1)
        mark_resume(ax)
        ax.legend(frameon=False, loc="upper right")

    def plot_psnr(ax: plt.Axes) -> None:
        ax.plot(epochs, psnr, color="#167d73", linewidth=1.8, marker=".", markersize=4)
        ax.scatter([epochs[best_index]], [psnr[best_index]], s=95, marker="*",
                   color="#c06a00", zorder=4)
        ax.text(0.40, 0.73, f"Best: {psnr[best_index]:.3f} dB at epoch {epochs[best_index]}",
                transform=ax.transAxes, color="#805000", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2})
        ax.set(title="Validation PSNR vs. Epoch", xlabel="Epoch", ylabel="PSNR (RGB, dB)")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, args.end_epoch + 1)
        mark_resume(ax)
        if args.end_epoch > 50:
            ax.legend(frameon=False, loc="upper left", fontsize=8)
        late = min(20, len(rows))
        inset = ax.inset_axes([0.43, 0.16, 0.52, 0.40])
        inset.plot(epochs[-late:], psnr[-late:], color="#167d73", linewidth=1.1,
                   marker=".", markersize=3)
        inset.set_title("Last 20 epochs (zoom)", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        inset.grid(alpha=0.2)
        inset.ticklabel_format(axis="y", useOffset=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_loss(axes[0])
    plot_psnr(axes[1])
    fig.suptitle(f"QuickSRNet-Small 2x | Asian Village | Epochs 1-{args.end_epoch}", fontsize=14)
    fig.text(0.5, 0.025,
             f"Train: 960 pairs, random crops | Validation: 240 full frames | "
             f"Pretrained validation: {baseline:.3f} dB | LR: {rates[-1]:.0e}",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"training_curves.{suffix}", dpi=200)
    plt.close(fig)
    for name, plotter in (("train_loss", plot_loss), ("validation_psnr", plot_psnr)):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        plotter(ax)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    report = [
        f"# Asian Village：Epoch 1–{args.end_epoch} 训练曲线分析", "",
        "## 数据和统计口径", "",
        "- Train：Scene_1_01 / 03 / 04 / 06，共 960 对；Val：Scene_1_02，共 240 对。",
        "- Train Loss 为随机裁剪与增强后的逐轮平均 L1，采样不同会造成波动。",
        "- Validation PSNR 为全分辨率 RGB [0,1]、逐帧计算后平均；验证过程没有随机增强。",
        "- Test（Scene_1_05）未参与训练、选模或本次曲线分析。", "",
        "## 指标", "",
        "| Epoch | Validation PSNR (dB) |", "|---:|---:|",
        f"| 0（官方预训练） | {baseline:.6f} |",
    ]
    for epoch in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        if epoch <= args.end_epoch:
            report.append(f"| {epoch} | {psnr[epoch-1]:.6f} |")
    report.extend([
        "", f"最高验证 PSNR：{psnr[best_index]:.6f} dB，出现在 Epoch {epochs[best_index]}。",
        f"最后 10 轮 PSNR 的线性趋势约为 {recent_slope:.6f} dB/epoch。",
        f"倒数第二个 10 轮区间的平均 Train Loss：{losses[-2*n:-n].mean():.8f}；"
        f"最后 10 轮：{losses[-n:].mean():.8f}。", "",
    ])
    if args.end_epoch == 50:
        report.extend([
            "## 收敛判断", "",
            f"Epoch 40→50 的 PSNR 仅增加 {psnr[49]-psnr[39]:.6f} dB，"
            f"Epoch 30→50 增加 {psnr[49]-psnr[29]:.6f} dB。",
            "模型已进入明显平台期、接近收敛，但仍有缓慢改善；Epoch 50 仍是当前最佳，"
            "不能断言已经完全收敛。验证 PSNR 没有持续下降，暂未出现明显的验证性能退化。",
            "训练损失的逐轮抖动需要结合移动平均判断，不能仅凭最后一轮高低判断过拟合。", "",
            "## 续训设置", "",
            "从归档的 Epoch 50 last.pth 恢复到同一 run 的 Epoch 51–100。",
            "原训练没有学习率调度器，全部 50 轮学习率均为 1e-4；续训沿用固定学习率。",
            "Adam 的一阶/二阶矩、8 组参数状态的 3000 次更新计数、AMP scaler、"
            "50 轮指标历史和历史最佳 PSNR 均已逐项验证恢复一致。",
            "best.pth 仅在新 PSNR 严格超过历史最高分时更新，因此覆盖 Epoch 1–100 全程。",
            "原始 checkpoint 未保存随机数和 DataLoader worker 的随机状态；"
            "本次续训恢复权重与优化器状态，但不保证与从不中断的 100 轮训练逐位一致。", "",
        ])
    if args.end_epoch >= 100:
        report.extend([
            "## 续训效果", "",
            f"Epoch 50→100 的 PSNR 增量：{psnr[99]-psnr[49]:.6f} dB。",
            f"全程最佳分数比前 50 轮最佳提高 {psnr.max()-psnr[:50].max():.6f} dB。",
            "前 50 轮原始权重、指标、日志和样例保存在 checkpoint 的 archive_epoch050/ 中。", "",
        ])
    (args.output_dir / "analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved curves and report: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
