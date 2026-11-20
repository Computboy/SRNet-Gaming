"""Regenerate the training curves and test crops for Sections 3.5 and 4."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset
from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol, masked_rgb_psnr_per_image
from src.metrics import rgb_psnr_per_image
from src.trainer import build_quicksrnet_from_local_checkpoint

PAPER = ROOT / "outputs/paper_chapters_3_5_4"
EPOCHS = np.arange(1, 151)
BENCHMARK_CANDIDATES = (
    ROOT / "outputs/experiments/medium-edge-l1-lambda-0p5-001/summary/benchmark_candidates.csv"
)

# Regions chosen from the HR frames. Coordinates are zero-based HR xyxy.
RECONSTRUCTION_REGIONS = {
    10: ("Character silhouette", (892, 366, 1148, 622)),
    50: ("Roof silhouette against sky", (508, 194, 764, 450)),
    70: ("Stone steps and carved ramp", (1166, 540, 1422, 796)),
    160: ("Masonry corner", (674, 334, 930, 590)),
    220: ("Paving joints and moss", (680, 784, 936, 1040)),
}
SMALL_EDGE_REGIONS = {
    10: ("Carved stair ramp", (304, 388, 560, 644)),
    70: ("Eaves against sky", (392, 120, 648, 376)),
    220: ("Character and foliage", (794, 388, 1050, 644)),
}
RECONSTRUCTION_KEYS = ("small_l1", "medium_l1", "medium_edge")
SMALL_EDGE_KEYS = ("small_l1", "small_edge_01", "small_edge")


def plot_curves(evaluation, report, destination):
    colors = ("#2166ac", "#c45a1a")
    labels = (report.loss_label(0), report.loss_label(0.5))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="col")

    for row, model in enumerate(("Medium", "Small")):
        loss_ax, psnr_ax = axes[row]
        total = loss_ax.inset_axes((.53, .49, .42, .36))
        late = psnr_ax.inset_axes((.51, .14, .44, .37))
        for key, color, label in zip(("l1", "edge"), colors, labels, strict=True):
            record = evaluation["checkpoints"][f"{model.lower()}_{key}"]
            experiment = report.load_run(record["run"], len(EPOCHS))
            history = experiment["history"]
            rgb = np.array([float(point["train_global_l1"]) for point in history])
            objective = np.array([float(point["train_total_loss"]) for point in history])
            psnr = np.array([float(point["val_psnr_db"]) for point in history])
            if not all(np.isfinite(values).all() for values in (rgb, objective, psnr)):
                raise ValueError(f"Non-finite epoch metrics in {record['run']}")

            best = int(record["selected_epoch"])
            if best != experiment["summary"]["selected"]["psnr"]["epoch"]:
                raise ValueError(f"PSNR checkpoint differs from the training history: {record['run']}")

            loss_ax.plot(EPOCHS, rgb, color=color, alpha=.28, linewidth=1.0)
            smoothed = np.convolve(rgb, np.ones(5) / 5, mode="valid")
            loss_ax.plot(EPOCHS[4:], smoothed, color=color, linewidth=1.6, label=label)
            total.plot(EPOCHS, objective, color=color, linewidth=1.1)
            psnr_ax.plot(EPOCHS, psnr, color=color, linewidth=1.55)
            psnr_ax.scatter(best, psnr[best - 1], edgecolors=color, facecolors="none",
                            linewidths=1.3, s=45, zorder=4)
            late.plot(EPOCHS[-20:], psnr[-20:], color=color, linewidth=1.05,
                      marker=".", markersize=3)
            late.scatter(best, psnr[best - 1], edgecolors=color, facecolors="none",
                         linewidths=1.1, s=28, zorder=4)

        loss_ax.set(title=f"{model} | Training loss", ylabel="Mean training RGB L1")
        psnr_ax.set(title=f"{model} | Validation PSNR", ylabel="Full-image PSNR (dB)")
        for ax in (loss_ax, psnr_ax):
            ax.set(xlabel="Epoch", xlim=(0, 153), xticks=[0, 25, 50, 75, 100, 125, 150])
            ax.grid(alpha=.2)
        for ax, title in ((total, "Total training loss"), (late, "Last 20 epochs (zoom)")):
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(alpha=.2)
        total.set(xlim=(0, 153), ylim=(.012, .045), xticks=[0, 50, 100, 150])
        late.set(xlim=(130, 151), xticks=[131, 135, 140, 145, 150])

    fig.suptitle("QuickSRNet 2× | Asian Village | Epochs 1–150", fontsize=14)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"figure_1_training_curves.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def frame_strip(panels, title, renderer):
    gap = 12
    width, height = panels[0][0].size
    canvas = Image.new("RGB", (len(panels) * (width + gap) + gap, height + 108 + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), title, fill="#172033", font=renderer.font(21, True))
    for index, (image, label, caption, highlighted) in enumerate(panels):
        left = gap + index * (width + gap)
        draw.text((left, 42), label, fill="#006b60" if highlighted else "#172033",
                  font=renderer.font(20, highlighted))
        draw.text((left, 70), caption, fill="#526071", font=renderer.font(17))
        canvas.paste(image, (left, 108))
    return canvas


def load_models(records, renderer, device):
    models = {}
    for key, record in records.items():
        path = Path(record["checkpoint"])
        if renderer.fingerprint(path)["sha256"] != record["sha256"]:
            raise ValueError(f"Checkpoint differs from the recorded candidate: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["epoch"] != int(record["selected_epoch"]) or saved["run_name"] != record["run"]:
            raise ValueError(f"Checkpoint metadata differs from the recorded candidate: {path}")
        config = saved["config"]["model"]
        model = build_quicksrnet_from_local_checkpoint(
            ROOT / config["pretrained_checkpoint"], model_name=config["name"]
        )
        model.model.load_state_dict(saved["state_dict"], strict=True)
        models[key] = model.eval().float().to(device)
    return models


def crop_prediction(prediction, box, renderer):
    # Match the chapter's original export rounding for predicted pixels.
    pixels = np.rint(renderer.hwc(prediction) * 255).astype(np.uint8)
    return Image.fromarray(pixels).crop(box)


def reconstruction_strip(frame, region, box, reference, lr, hr, predictions, records, scores, renderer):
    lr_display = renderer.image_from_float(renderer.hwc(lr)).resize(reference.size, Image.Resampling.NEAREST)
    panels = [(reference.crop(box), "HR Ground Truth", "Reference RGB image", False),
              (lr_display.crop(box), "LR (display only)", "Nearest display", False)]
    for key in RECONSTRUCTION_KEYS:
        prediction = predictions[key]
        psnr = float(scores[f"{key}_psnr_db"])
        if not math.isclose(rgb_psnr_per_image(prediction, hr).item(), psnr, abs_tol=1e-5):
            raise ValueError(f"Reconstruction does not match the saved PSNR: {frame}/{key}")
        ssim = float(scores[f"{key}_ssim"])
        record = records[key]
        with_edge = key == "medium_edge"
        name = "Medium + Edge-L1" if with_edge else record["model"]
        image = crop_prediction(prediction, box, renderer)
        panels.append((image, f"{name} (E{record['selected_epoch']})",
                       f"PSNR {psnr:.3f} dB | SSIM {ssim:.4f}", with_edge))
    return frame_strip(panels, f"Scene_1_05_frame_{frame:04d} | {region} | Identical HR crop", renderer)


def small_edge_strip(frame, region, box, reference, hr, predictions, records, edge_metric, renderer):
    x0, y0, x1, y1 = box
    crop = (..., slice(y0, y1), slice(x0, x1))
    mask = edge_metric.mask(hr)[crop]
    panels = [(reference.crop(box), "HR Ground Truth", "Reference RGB image", False)]
    for key in SMALL_EDGE_KEYS:
        prediction = predictions[key]
        record = records[key]
        weight = float(record["edge_l1_weight"])
        psnr = masked_rgb_psnr_per_image(prediction[crop], hr[crop], mask)["psnr_db"].item()
        image = crop_prediction(prediction, box, renderer)
        panels.append((image, f"Small, λ = {weight:g} (E{record['selected_epoch']})",
                       f"ROI edge PSNR {psnr:.3f} dB", weight == 0.5))
    return frame_strip(panels, f"Scene_1_05_frame_{frame:04d} | {region} | Identical HR crop", renderer)


def render_reconstruction(evaluation, scores, small_edge_01, renderer, destination):
    scene = evaluation["scene"]
    if scene != "Scene_1_05":
        raise ValueError(f"Unexpected test scene: {scene}")
    data = ROOT / "data/gaming/test"
    dataset = PairedImageDataset(data / "LR", data / "HR")
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)

    records = {**evaluation["checkpoints"], "small_edge_01": small_edge_01}
    models = load_models(records, renderer, device)
    fields = EdgeValidationProtocol.__dataclass_fields__
    protocol = EdgeValidationProtocol(**{key: evaluation["edge_protocol"][key] for key in fields})
    edge_metric = EdgeRegionMetric(protocol).to(device)

    strips, small_strips = [], []
    with torch.inference_mode():
        for frame, (region, box) in RECONSTRUCTION_REGIONS.items():
            sample = dataset[frame]
            if sample["name"] != f"{scene.lower()}/frame_{frame:04d}":
                raise ValueError(f"Unexpected test pair: {sample['name']}")
            lr, hr = (sample[key].unsqueeze(0).to(device) for key in ("lr", "hr"))
            reference = renderer.image_from_float(renderer.hwc(hr))
            row = scores[f"{frame:04d}"]
            predictions = {key: models[key](lr).clamp(0, 1) for key in RECONSTRUCTION_KEYS}
            strips.append(reconstruction_strip(frame, region, box, reference, lr, hr,
                                               predictions, records, row, renderer))
            if frame in SMALL_EDGE_REGIONS:
                small_region, small_box = SMALL_EDGE_REGIONS[frame]
                predictions.update({key: models[key](lr).clamp(0, 1) for key in SMALL_EDGE_KEYS[1:]})
                small_strips.append(small_edge_strip(frame, small_region, small_box, reference,
                                                    hr, predictions, records, edge_metric, renderer))
    renderer.save_montage(strips, destination / "figure_2_test_reconstruction.png")
    renderer.save_montage(small_strips, destination / "figure_small_edge_weight.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER)
    parser.add_argument("--crops-only", action="store_true", help="Regenerate the two ROI comparisons.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = import_module("11_compare_edge_l1_ab")
    renderer = import_module("05_verify_validation")
    evidence = PAPER / "evidence"
    evaluation = json.loads((evidence / "scene_1_05_summary.json").read_text(encoding="utf-8"))
    scores = {row["frame"]: row for row in report.read_csv(evidence / "scene_1_05_per_frame.csv")}
    [small_edge_01] = [
        candidate for candidate in report.read_csv(BENCHMARK_CANDIDATES)
        if candidate["model"] == "Small" and float(candidate["edge_l1_weight"]) == 0.1
    ]
    if not args.crops_only:
        plot_curves(evaluation, report, args.output_dir)
    render_reconstruction(evaluation, scores, small_edge_01, renderer, args.output_dir)


if __name__ == "__main__":
    main()
