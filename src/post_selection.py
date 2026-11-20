"""Report-only SSIM on complete validation images of already selected models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import PairedImageDataset
from src.edge_validation import EdgeRegionMetric, EdgeValidationProtocol
from src.metrics import SSIM_PARAMETERS, rgb_psnr_per_image, rgb_ssim


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(dataset, manifest):
    pairs = [{"key": pair.key, "lr_sha256": sha256(pair.lr_path), "hr_sha256": sha256(pair.hr_path)}
             for pair in dataset.pairs]
    fingerprint = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode("utf-8")).hexdigest()
    if pairs != manifest["pairs"] or fingerprint != manifest["sha256"]:
        raise ValueError("Validation image contents changed before post-selection SSIM")


def evaluate_rows(model, loader, device, protocol, expected_keys):
    """Same FP32 inference and metric formulas as per-epoch validation; no updates."""
    model = model.to(device).float().eval()
    edge_metric = EdgeRegionMetric(protocol).to(device)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            output = model(lr)
            if output.shape != hr.shape or not torch.isfinite(output).all():
                raise ValueError("Invalid post-selection inference output")
            psnr = rgb_psnr_per_image(output, hr)
            edge = edge_metric(output, hr)
            prediction = output.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            reference = hr.permute(0, 2, 3, 1).cpu().numpy()
            for index, key in enumerate(batch["name"]):
                rows.append({"key": key, "psnr_db": psnr[index].item(),
                             "ssim": rgb_ssim(prediction[index], reference[index]),
                             "edge_region_psnr_db": edge["psnr_db"][index].item()
                             if edge["pixels"][index].item() > 0 else None})
            if len(rows) == 1 or len(rows) % 60 == 0:
                print(f"Post-selection validation: {len(rows)}/{len(expected_keys)} images", flush=True)
    if len(rows) != len(expected_keys) or {row["key"] for row in rows} != set(expected_keys):
        raise ValueError("Post-selection SSIM must cover the complete validation manifest exactly once")
    return rows


def merge_reported_ssim(original, evaluated):
    """Verify selection metrics, then add SSIM to a separate reporting copy."""
    lookup = {row["key"]: row for row in evaluated}
    if len(lookup) != len(evaluated) or len(original) != len(evaluated) or set(lookup) != {r["key"] for r in original}:
        raise ValueError("Post-selection frame keys do not match the selected epoch CSV")
    result = []
    for row in original:
        score = lookup[row["key"]]
        for metric in ("psnr_db", "edge_region_psnr_db", "ssim"):
            before, after = row.get(metric), score[metric]
            if before in ("", None):
                if metric != "ssim" and after is not None:
                    raise ValueError(f"Empty-mask result changed for {row['key']}/{metric}")
                continue
            if after is None or not math.isclose(float(before), after, abs_tol=1e-5, rel_tol=1e-7):
                raise ValueError(f"Post-selection score differs from saved validation: {row['key']}/{metric}")
        result.append({**row, "ssim": score["ssim"]})
    return result


def selected_ssim(data, root, output, read_csv, save_csv, model_factory):
    """Evaluate each distinct selected epoch once; preserve all training artifacts."""
    config = data["checkpoint"]["config"]
    dataset = PairedImageDataset(root / config["data"]["val_lr"], root / config["data"]["val_hr"])
    expected = {pair["key"] for pair in data["manifest"]["pairs"]}
    if len(expected) != 240 or any(not key.startswith("scene_1_02/") for key in expected):
        raise ValueError("Formal post-selection reporting is restricted to all 240 Scene_1_02 validation frames")
    verify_manifest(dataset, data["manifest"])
    output.mkdir(parents=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = EdgeValidationProtocol.from_config(config)
    pretrained = root / config["model"]["pretrained_checkpoint"]
    if sha256(pretrained) != config["runtime"]["pretrained_checkpoint_sha256"]:
        raise ValueError("Official initialization changed before post-selection reporting")
    model = model_factory(pretrained)
    evaluated = {}
    for strategy, choice in data["summary"]["selected"].items():
        epoch = int(choice["epoch"])
        checkpoint_path = Path(choice["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != epoch or checkpoint["validation_fingerprint"] != data["manifest"]["sha256"]:
            raise ValueError("Selected checkpoint metadata changed")
        if epoch in evaluated:
            if any(not torch.equal(checkpoint["state_dict"][name], tensor)
                   for name, tensor in evaluated[epoch]["state_dict"].items()):
                raise ValueError("Same selected epoch has inconsistent weights")
            continue
        print(f"SSIM after selection: {data['run']} / epoch {epoch}", flush=True)
        model.model.load_state_dict(checkpoint["state_dict"], strict=True)
        rows = evaluate_rows(model, loader, device, protocol, expected)
        original = read_csv(root / "outputs/metrics" / data["run"] / "validation" / f"epoch{epoch:03d}.csv")
        merged = merge_reported_ssim(original, rows)
        path = output / f"epoch{epoch:03d}.csv"
        save_csv(path, merged)
        metadata = {"stage": "post_selection", "run": data["run"], "epoch": epoch,
                    "frames": len(merged), "mean_ssim": sum(row["ssim"] for row in merged) / len(merged),
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                    "validation_fingerprint": data["manifest"]["sha256"], "ssim_parameters": SSIM_PARAMETERS,
                    "inference": "FP32; same full-image formulas; no change to selection", "test_accessed": False}
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        evaluated[epoch] = {"rows": merged, "metadata": metadata, "state_dict": checkpoint["state_dict"]}
    data["post_selection_ssim"] = evaluated
    return evaluated
