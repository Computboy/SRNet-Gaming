"""Run prepared C, then A/C analysis. Requires explicit --start-training."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.capacity import MEDIUM_WEIGHT_SHA256, sha256, verify_capacity_configs
from src.utils import load_yaml_config, project_path


def existing_runner():
    spec = importlib.util.spec_from_file_location("ab_runner_helpers", ROOT / "scripts/10_run_edge_l1_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/asian_village_medium_l1.yaml"))
    parser.add_argument("--preparation", type=Path, default=Path("outputs/experiments/medium-l1-preparation-001/preparation.json"))
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.start_training:
        parser.exit(0, "No training started. To run C explicitly, add --start-training after reviewing preparation.json.\n")
    helpers = existing_runner()
    config_path = project_path(ROOT, args.config)
    config = load_yaml_config(config_path)
    run = config["experiment"]["run_name"]
    baseline_run = config["experiment"]["baseline_run"]
    if Path(run).name != run or run in {".", "..", baseline_run, "asian-village-edge-l1-001"}:
        raise ValueError("C requires a new independent experiment name")
    job = project_path(ROOT, config["experiment"]["job_dir"])
    if not job.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("C job directory must stay inside outputs/experiments")
    baseline_path = ROOT / "checkpoints/finetuned" / baseline_run / "last.pth"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline["epoch"] != 100:
        raise ValueError("The comparison baseline A must have completed 100 epochs")
    verify_capacity_configs(baseline["config"], config)
    prepared = json.loads(project_path(ROOT, args.preparation).read_text(encoding="utf-8"))
    if prepared["C_run"] != run or prepared["baseline_run"] != baseline_run:
        raise ValueError("Preparation belongs to a different capacity experiment")
    for name, expected in prepared["sources_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Prepared source/config changed; repeat preparation: {name}")
    if sha256(baseline_path) != prepared["baseline_checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint changed after preparation")
    pretrained = project_path(ROOT, config["model"]["pretrained_checkpoint"])
    if sha256(pretrained) != MEDIUM_WEIGHT_SHA256:
        raise ValueError("Official Medium weights changed after preparation")
    status_path = job / "status.json"
    if job.exists() and not args.resume:
        raise ValueError("C job exists; inspect it and use --resume explicitly")
    if args.resume and status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if helpers.process_alive(previous.get("runner_pid")) or helpers.process_alive(previous.get("child_pid")):
            raise ValueError("A C runner or training child is still active")
    if not args.resume:
        for key in ("checkpoint_dir", "log_dir", "metric_dir", "sample_dir"):
            if (project_path(ROOT, config["output"][key]) / run).exists():
                raise ValueError(f"C output already exists: {key}; refusing overwrite")
    job.mkdir(parents=True, exist_ok=True)
    status = {"phase": "preflight", "started_at": helpers.now(), "runner_pid": os.getpid(),
              "child_pid": None, "run": run, "baseline_run": baseline_run, "completed_epochs": 0,
              "target_epochs": 100, "test_accessed": False}
    helpers.write_json(status_path, status)
    guard = helpers.IdleSleepGuard()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Declared C experiment requires CUDA")
        status["idle_sleep_inhibited"] = guard.start()
        images = helpers.image_hashes(config)
        old = json.loads((ROOT / "outputs/experiments/edge-l1-ab-002-ssim-posthoc/frozen_manifest.json").read_text(encoding="utf-8"))
        if images != old["train_validation_images"]:
            raise ValueError("Train/validation data differ from A")
        source_hashes = {**prepared["sources_sha256"], str(pretrained.relative_to(ROOT)): sha256(pretrained),
                         str(baseline_path.relative_to(ROOT)): sha256(baseline_path)}
        manifest = {"sources": source_hashes, "train_validation_images": images}
        manifest_path = job / "frozen_manifest.json"
        if args.resume:
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("C source, initialization or data changed since launch")
        else:
            helpers.write_json(manifest_path, manifest)
            shutil.copyfile(project_path(ROOT, args.preparation), job / "preparation.json")
            for name in prepared["sources_sha256"]:
                target = job / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
        completed = helpers.inspect_run(run, config, 100)
        if completed < 100:
            command = [sys.executable, "-u", str(ROOT / "scripts/03_finetune.py"), "--config", str(config_path), "--epochs", "100", "--device", "cuda"]
            command += ["--resume", str(ROOT / "checkpoints/finetuned" / run / "last.pth")] if completed else ["--experiment-name", run]
            status.update(phase="training_C", command=command, completed_epochs=completed)
            with (job / "train_C.stdout.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                status["child_pid"] = child.pid
                helpers.write_json(status_path, status)
                while child.poll() is None:
                    status.update(helpers.progress(run), updated_at=helpers.now())
                    helpers.write_json(status_path, status)
                    time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"C training failed with exit code {child.returncode}")
        if helpers.inspect_run(run, config, 100) != 100:
            raise ValueError("C did not complete the declared budget")
        for name, expected in source_hashes.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Source changed during C training: {name}")
        status.update(phase="analysis", child_pid=None, completed_epochs=100, updated_at=helpers.now())
        helpers.write_json(status_path, status)
        with (job / "analysis.stdout.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/15_compare_capacity.py"),
                            "--baseline-run", baseline_run, "--candidate-run", run,
                            "--output-dir", str(job / "analysis")], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        status.update(phase="complete", finished_at=helpers.now(), updated_at=helpers.now(), analysis_path=str(job / "analysis"))
        helpers.write_json(status_path, status)
    except BaseException as exc:
        status.update(phase="failed", error=str(exc), updated_at=helpers.now())
        helpers.write_json(status_path, status)
        raise
    finally:
        guard.close()


if __name__ == "__main__":
    main()
