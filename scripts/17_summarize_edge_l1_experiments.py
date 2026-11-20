"""Summarize five completed QuickSRNet runs using validation-selected checkpoints."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs/experiments"
SMALL_BASE = EXPERIMENTS / "edge-l1-ab-004-150-epochs-amp-recovery"
SMALL_HIGH_WEIGHT = EXPERIMENTS / "small-edge-l1-lambda-0p5-001"
MEDIUM = EXPERIMENTS / "medium-edge-l1-lambda-0p5-001"
METRICS = ("val_psnr_db", "val_ssim", "val_edge_region_psnr_db", "val_loss", "val_weighted_score")


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect(report, model, weight, job, label, run):
    analysis = job / "analysis"
    status = json.loads((job / "status.json").read_text(encoding="utf-8"))
    if status["phase"] != "complete" or status["target_epochs_per_run"] != 150:
        raise ValueError(f"Expected a completed 150-epoch comparison: {job}")
    run_data = report["load_run"](run, 150)
    config = run_data["checkpoint"]["config"]
    if config["model"]["name"] != f"quicksrnet_{model.lower()}" or config["training"].get("edge_l1_weight", 0.0) != weight:
        raise ValueError(f"Unexpected model or edge weight in {run}")
    [selected] = [row for row in report["read_csv"](analysis / "selected_models.csv")
                  if row["run"] == run and row["experiment"] == label and row["strategy"] == "psnr"]
    epoch = int(selected["epoch"])
    epoch_metrics = run_data["history"][epoch - 1]
    winner = run_data["summary"]["selected"]["psnr"]
    if epoch != winner["epoch"] or not math.isclose(
        float(selected["val_psnr_db"]), float(epoch_metrics["val_psnr_db"]), abs_tol=1e-8
    ):
        raise ValueError(f"Selected checkpoint differs from the saved PSNR history: {run}")
    checkpoint_dir = ROOT / "checkpoints/finetuned" / run
    artifact = checkpoint_dir / "epochs" / f"epoch{epoch:03d}.pth"
    artifact_hash = checksum(artifact)
    if artifact_hash != checksum(Path(selected["checkpoint"])):
        raise ValueError(f"Selected checkpoint differs from its archived epoch: {run}")
    evidence = analysis / "post_selection_ssim" / label / f"epoch{epoch:03d}.json"
    evaluation = json.loads(evidence.read_text(encoding="utf-8"))
    if (evaluation["frames"] != 240 or evaluation["checkpoint_sha256"] != artifact_hash
            or not math.isclose(float(selected["val_ssim"]), evaluation["mean_ssim"], abs_tol=1e-10)
            or evaluation["validation_fingerprint"] != run_data["manifest"]["sha256"]):
        raise ValueError(f"Missing matching full-validation SSIM evidence: {run}")
    scores = {key: float(selected[key] if key in selected else epoch_metrics[key]) for key in METRICS}
    result = {"model": model, "edge_l1_weight": weight, "run": run, "epochs_trained": 150,
              "selected_epoch": epoch, **scores}
    candidate = {"model": model, "edge_l1_weight": weight, "run": run, "selected_epoch": epoch,
                 "checkpoint": artifact.as_posix(), "sha256": artifact_hash,
                 "last_checkpoint": (checkpoint_dir / "last.pth").as_posix()}
    return result, candidate, run_data["manifest"]["sha256"], run_data["summary"]["protocol"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium-job", type=Path, default=MEDIUM)
    args = parser.parse_args()
    medium = args.medium_job.resolve()
    report = runpy.run_path(str(ROOT / "scripts/11_compare_edge_l1_ab.py"))
    get = partial(collect, report)
    inputs = (
        ("Small", 0.0, SMALL_BASE, "A", "asian-village-selection-001"),
        ("Small", 0.1, SMALL_BASE, "B", "asian-village-edge-l1-001"),
        ("Small", 0.5, SMALL_HIGH_WEIGHT, "B", "asian-village-small-edge-l1-lambda-0p5-001"),
        ("Medium", 0.0, medium, "A", "asian-village-medium-l1-150-001"),
        ("Medium", 0.5, medium, "B", "asian-village-medium-edge-l1-lambda-0p5-001"),
    )
    rows, candidates, fingerprints, protocols = zip(*(get(*experiment) for experiment in inputs))
    if len(set(fingerprints)) != 1 or any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("Compared runs used different validation images or edge definitions")
    baselines = {row["model"]: row for row in rows if row["edge_l1_weight"] == 0}
    for row in rows:
        baseline = baselines[row["model"]]
        row.update({f"delta_{key}": row[key] - baseline[key] for key in METRICS})
    output = medium / "summary"
    output.mkdir(exist_ok=False)
    report["save_csv"](output / "five_runs.csv", rows)
    report["save_csv"](output / "benchmark_candidates.csv", candidates)

    lines = [
        "# QuickSRNet 的五组微调结果", "",
        "五组均训练 150 轮，按最高整帧验证 PSNR 选取 checkpoint。指标来自同一验证序列的完整 240 帧。",
        "增量以同架构的纯 L1 组为基准。PSNR 单位为 dB。", "",
        "| 模型 | λ | 入选轮次 | 整帧 PSNR | Edge-region PSNR | SSIM | Δ整帧 PSNR | ΔEdge-region PSNR | ΔSSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['edge_l1_weight']:g} | {row['selected_epoch']} | "
            f"{row['val_psnr_db']:.6f} | {row['val_edge_region_psnr_db']:.6f} | {row['val_ssim']:.6f} | "
            f"{row['delta_val_psnr_db']:+.6f} | {row['delta_val_edge_region_psnr_db']:+.6f} | {row['delta_val_ssim']:+.6f} |"
        )
    small, medium_edge = rows[2], rows[4]
    edge_gap = medium_edge["delta_val_edge_region_psnr_db"] - small["delta_val_edge_region_psnr_db"]
    lines.extend([
        "", "## 容量与边缘加权", "",
        f"λ=0.5 相对各自纯 L1 基线的边缘 PSNR 增量为 Small {small['delta_val_edge_region_psnr_db']:+.4f} dB、"
        f"Medium {medium_edge['delta_val_edge_region_psnr_db']:+.4f} dB。Medium 的增量减去 Small 的增量为 {edge_gap:+.4f} dB。",
        f"两者的整帧 PSNR 变化分别为 {small['delta_val_psnr_db']:+.4f} dB 和 {medium_edge['delta_val_psnr_db']:+.4f} dB，"
        f"SSIM 变化分别为 {small['delta_val_ssim']:+.6f} 和 {medium_edge['delta_val_ssim']:+.6f}。",
        "以上差异的适用范围是当前一个验证场景、一个随机种子、各架构的官方初始化和实际续训轨迹。",
        "", "[完整指标和差值](five_runs.csv) · [Benchmark 候选权重](benchmark_candidates.csv) · "
        "[Medium A/B 对照](../analysis/README.md)",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
