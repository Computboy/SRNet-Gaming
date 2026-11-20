"""Prepare C and run bounded smoke checks. Never starts formal training."""
from __future__ import annotations
import argparse
import importlib.util
from importlib.metadata import version
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, Subset
from qai_hub_models.models.quicksrnetmedium import model as official
from src.capacity import (MEDIUM_WEIGHT_SHA256, MEDIUM_WEIGHT_URL, benchmark_pair,
                          model_description, sha256, verify_capacity_configs)
from src.dataset import PairedImageDataset
from src.post_selection import verify_manifest
from src.trainer import (Trainer, build_optimizer, build_quicksrnet_from_local_checkpoint,
                         verify_checkpoint_reload)
from src.utils import load_yaml_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asian_village_medium_l1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/medium-l1-preparation-001")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "preparation.json").exists():
        raise ValueError("Preparation report exists; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(args.config)
    baseline_run = config["experiment"]["baseline_run"]
    baseline_path = ROOT / "checkpoints/finetuned" / baseline_run / "last.pth"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline["epoch"] != 100 or not baseline["validation_is_independent"]:
        raise ValueError("A must be a completed formal 100-epoch baseline")
    comparison = verify_capacity_configs(baseline["config"], config)
    seed_everything(config["seed"])
    pretrained = ROOT / config["model"]["pretrained_checkpoint"]
    if sha256(pretrained) != MEDIUM_WEIGHT_SHA256:
        raise ValueError("Official Medium checksum mismatch")
    assert official.NUM_CHANNELS == 32 and official.NUM_INTERMEDIATE_LAYERS == 5
    assert official.BASE_ASSET_URL.format(scale_factor=2) == MEDIUM_WEIGHT_URL
    medium = build_quicksrnet_from_local_checkpoint(pretrained, model_name="quicksrnet_medium")
    weights = torch.load(pretrained, map_location="cpu", weights_only=False)["state_dict"]
    assert set(medium.model.state_dict()) == set(weights)
    assert all(torch.equal(value, weights[name]) for name, value in medium.model.state_dict().items())
    assert isinstance(medium, official.QuickSRNetMedium)
    description = model_description(medium)
    assert description["parameters"] == 50604 and description["conv_layers"] == 7
    original = {key: value.clone() for key, value in medium.model.state_dict().items()}
    print("Official Medium f32-m5 loaded strictly: 50,604 parameters", flush=True)
    device = torch.device("cuda")
    logger = logging.getLogger("medium-preparation")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    trainer = Trainer(model=medium, optimizer=build_optimizer(medium, "adam", 1e-4), device=device,
        epochs=100, scale_factor=2, config=config, run_name="medium-preparation-smoke-only",
        pretrained_checkpoint=pretrained, checkpoint_dir=out / "smoke_checkpoints",
        metric_dir=out / "smoke_metrics", sample_dir=out / "smoke_samples", logger=logger,
        validation_is_independent=False, amp=True, save_samples=False)
    synthetic = [{"lr": torch.rand(3, 128, 128), "hr": torch.rand(3, 256, 256), "name": f"synthetic-{i}"}
                 for i in range(16)]
    torch.cuda.reset_peak_memory_stats()
    loss = trainer.train_one_epoch(DataLoader(synthetic, batch_size=16), 0)
    peak = torch.cuda.max_memory_allocated()
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    smoke_checkpoint = trainer.save_checkpoint("synthetic_one_step.pth", 0)
    difference = verify_checkpoint_reload(checkpoint_path=smoke_checkpoint, pretrained_checkpoint=pretrained,
        scale_factor=2, trained_model=medium, sample_lr=synthetic[0]["lr"].unsqueeze(0), device=device)
    medium.model.load_state_dict(original, strict=True)
    print("One synthetic AMP/Adam step at batch=16, patch=128 passed; official weights restored for validation", flush=True)
    dataset = PairedImageDataset(ROOT / config["data"]["val_lr"], ROOT / config["data"]["val_hr"])
    manifest = json.loads((ROOT / "outputs/logs" / baseline_run / "validation_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(dataset, manifest)
    assert len(dataset) == 240
    frames = [0, 120, 239]
    validation_loss, validation_psnr = trainer.validate(DataLoader(Subset(dataset, frames), batch_size=1), 0)
    assert trainer.last_validation_details["val_ssim"] is None
    assert trainer.last_validation_details["val_edge_images"] == 3
    baseline_model = build_quicksrnet_from_local_checkpoint(ROOT / baseline["config"]["model"]["pretrained_checkpoint"])
    baseline_model.model.load_state_dict(baseline["state_dict"], strict=True)
    sample = dataset[0]["lr"].unsqueeze(0).to(device)
    with torch.inference_mode():
        shape = list(medium(sample).shape)
    assert shape == [1, 3, 1080, 1920]
    timing = benchmark_pair({"A": baseline_model, "C": medium}, sample, warmup=5, iterations=10, rounds=2)
    timing["stage"] = "preparation only; A fine-tuned, C official pretrained; remeasure both selected models after C training"
    (out / "latency_smoke.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    entry_spec = importlib.util.spec_from_file_location("existing_ab_runner", ROOT / "scripts/10_run_edge_l1_ab.py")
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    images = entry.image_hashes(config)
    old = json.loads((ROOT / "outputs/experiments/edge-l1-ab-002-ssim-posthoc/frozen_manifest.json").read_text(encoding="utf-8"))
    assert images == old["train_validation_images"]
    preserved = None
    inventory = out / "existing_artifacts_sha256.json"
    if inventory.exists():
        saved = json.loads(inventory.read_text(encoding="utf-8"))
        for name, expected in saved.items():
            assert sha256(ROOT / name) == expected, f"Existing artifact modified: {name}"
        preserved = len(saved)
    sources = ["src/trainer.py", "src/capacity.py", "src/edge_validation.py", "src/metrics.py", "src/losses.py",
               "src/post_selection.py", "src/dataset.py", "src/utils.py", "scripts/03_finetune.py",
               "scripts/09_plot_selection_comparison.py", "scripts/10_run_edge_l1_ab.py", "scripts/11_compare_edge_l1_ab.py",
               "scripts/13_prepare_medium_l1.py", "scripts/14_run_medium_l1.py", "scripts/15_compare_capacity.py",
               baseline["config"]["model"]["pretrained_checkpoint"], str(args.config.resolve().relative_to(ROOT))]
    report = {"state": "prepared_not_started", "formal_epochs_completed": 0, "formal_training_started": False,
              "C_run": config["experiment"]["run_name"], "baseline_run": baseline_run,
              "official_medium": {**description, "weights": str(pretrained), "sha256": MEDIUM_WEIGHT_SHA256,
                                   "source_url": MEDIUM_WEIGHT_URL, "strict_load": True, "all_tensors_match": True,
                                   "installed_official_model_source_sha256": sha256(Path(official.__file__))},
              "environment": {"python": sys.version, "torch": str(torch.__version__),
                              "qai_hub_models": version("qai-hub-models"), "gpu": torch.cuda.get_device_name(0)},
              "fairness": comparison, "synthetic_one_step_loss": loss,
              "training_batch_size_checked": 16, "lr_patch_size_checked": 128,
              "peak_training_smoke_allocated_bytes": peak, "checkpoint_reload_difference": difference,
              "full_forward_output_shape": shape, "validation_manifest_frames": 240,
              "smoke_validation_frames": frames, "smoke_val_l1": validation_loss, "smoke_val_psnr": validation_psnr,
              "smoke_validation_details": trainer.last_validation_details,
              "train_validation_images_unchanged": True, "old_artifacts_unchanged": preserved,
              "sources_sha256": {name: sha256(ROOT / name) for name in sources},
              "baseline_checkpoint_sha256": sha256(baseline_path), "test_accessed": False}
    (out / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    source_record = {"source_url": MEDIUM_WEIGHT_URL, "sha256": MEDIUM_WEIGHT_SHA256, "bytes": pretrained.stat().st_size}
    pretrained.with_suffix(".source.json").write_text(json.dumps(source_record, indent=2), encoding="utf-8")
    print(json.dumps({"state": report["state"], "parameters": description["parameters"],
                      "latency_median_ms": {k: v["median_ms"] for k, v in timing["models"].items()},
                      "existing_artifacts_verified": preserved}, indent=2), flush=True)


if __name__ == "__main__":
    main()
