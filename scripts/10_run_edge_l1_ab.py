"""Run the predeclared A/B experiment serially, then generate validation analysis.

No Test access. An incomplete existing run resumes; a verified complete run is
reused. Source/config/data hashes are frozen before training, and failures stop B.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
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
from src.edge_validation import EdgeValidationProtocol, json_safe, select_epochs
from src.losses import TrainingLossProtocol
from src.utils import load_yaml_config, project_path
from src.metrics import ssim_schedule_from_config, selection_validation_settings


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_pair(a, b, *, a_weight=0.0, b_weight=0.1):
    for key in ("model", "data", "validation", "seed", "device", "output"):
        if a.get(key) != b.get(key):
            raise ValueError(f"A/B must have identical {key}")
    ta, tb = dict(a["training"]), dict(b["training"])
    for settings in (ta, tb):
        settings.pop("epochs", None)  # Both runs use the command-line epoch budget.
        settings.pop("loss", None)
        settings.pop("edge_l1_weight", None)
    if ta != tb:
        raise ValueError("A/B differ in training settings other than loss")
    for label, config, weight in (("A", a, a_weight), ("B", b, b_weight)):
        expected = TrainingLossProtocol("l1" if weight == 0 else "l1_edge", weight)
        if TrainingLossProtocol.from_config(config) != expected:
            raise ValueError(f"{label} must use the declared loss {expected.mode} with edge_l1_weight={weight:g}")
    if not EdgeValidationProtocol.from_config(a).enabled:
        raise ValueError("Both runs require the fixed Stage 1 edge validation protocol")
    ssim_schedule_from_config(a)


def verify_ssim_revision(old, new):
    """Only a user-declared report timing change may reuse the paused trajectory."""
    left, right = dict(old), dict(new)
    left["validation"] = selection_validation_settings(old)
    right["validation"] = selection_validation_settings(new)
    if left != right or ssim_schedule_from_config(new) != "post_selection":
        raise ValueError("The declared revision may change only the SSIM reporting schedule")


class IdleSleepGuard:
    """Keep the system awake only for this runner's lifetime; no power-plan edits."""
    def __init__(self):
        self.kernel = None
        self.active = False

    def start(self):
        if os.name == "nt":
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
            self.kernel.SetThreadExecutionState.restype = ctypes.c_uint
            self.active = bool(self.kernel.SetThreadExecutionState(0x80000001))
        return self.active

    def close(self):
        if self.active:
            self.kernel.SetThreadExecutionState(0x80000000)
            self.active = False


def image_hashes(config):
    result = {}
    for key in ("train_lr", "train_hr", "val_lr", "val_hr"):
        directory = project_path(ROOT, config["data"][key])
        if not directory.is_dir():
            raise ValueError(f"Missing input directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                result[str(path)] = digest(path)
    if not result:
        raise ValueError("No train/validation images")
    return result


def process_alive(pid):
    if not pid:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, int(pid))
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def progress(run):
    path = ROOT / "outputs/metrics" / run / "metrics.csv"
    if not path.exists():
        return {"completed_epochs": 0}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {"completed_epochs": int(rows[-1]["epoch"]), "latest_metrics": rows[-1]} if rows else {"completed_epochs": 0}


def inspect_run(run, config, target_epochs):
    directory = ROOT / "checkpoints/finetuned" / run
    last = directory / "last.pth"
    if not last.exists():
        if directory.exists():
            raise ValueError(f"Existing incomplete output without last.pth requires inspection: {directory}")
        return 0
    checkpoint = torch.load(last, map_location="cpu", weights_only=True)
    if not checkpoint.get("validation_is_independent") or checkpoint.get("run_name") != run:
        raise ValueError(f"Not the expected independent formal run: {run}")
    protocol = EdgeValidationProtocol.from_config(config)
    if checkpoint.get("edge_validation_protocol") != protocol.definition():
        raise ValueError(f"Existing run has a different validation protocol: {run}")
    if checkpoint.get("training_loss_protocol") != TrainingLossProtocol.from_config(config).definition(protocol):
        raise ValueError(f"Existing run has a different or unrecorded loss protocol: {run}")
    if selection_validation_settings(checkpoint["config"]) != selection_validation_settings(config):
        raise ValueError(f"Existing run has different selection validation settings: {run}")
    for key in ("model", "data", "seed"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Existing run differs in {key}: {run}")
    saved_training = dict(checkpoint["config"]["training"])
    expected_training = dict(config["training"])
    saved_training.pop("epochs", None)
    expected_training.pop("epochs", None)
    if saved_training != expected_training:
        raise ValueError(f"Existing run has different training hyperparameters: {run}")
    pretrained_hash = digest(project_path(ROOT, config["model"]["pretrained_checkpoint"]))
    if checkpoint["config"].get("runtime", {}).get("pretrained_checkpoint_sha256") != pretrained_hash:
        raise ValueError(f"Cannot verify the same official initialization for {run}")
    epoch = int(checkpoint["epoch"])
    if epoch > target_epochs:
        raise ValueError(f"Existing run exceeds the declared epoch budget: {run}")
    if [int(row["epoch"]) for row in checkpoint["metrics"]] != list(range(1, epoch + 1)):
        raise ValueError(f"Incomplete epoch history: {run}")
    if epoch == target_epochs:
        summary = json.loads((ROOT / "outputs/metrics" / run / "selection_summary.json").read_text(encoding="utf-8"))
        expected = select_epochs(checkpoint["metrics"], protocol)
        for strategy, choice in expected.items():
            if summary["selected"][strategy]["epoch"] != choice["epoch"]:
                raise ValueError(f"Selection summary mismatch: {run}/{strategy}")
            if not Path(summary["selected"][strategy]["checkpoint"]).is_file():
                raise ValueError(f"Missing selected checkpoint: {run}/{strategy}")
        for index in range(1, epoch + 1):
            if not (directory / "epochs" / f"epoch{index:03d}.pth").is_file():
                raise ValueError(f"Missing retained checkpoint: {run}/{index}")
    return epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-config", type=Path, default=Path("configs/asian_village_selection.yaml"))
    parser.add_argument("--b-config", type=Path, default=Path("configs/asian_village_edge_l1.yaml"))
    parser.add_argument("--a-run", default="asian-village-selection-001")
    parser.add_argument("--b-run", default="asian-village-edge-l1-001")
    parser.add_argument("--a-edge-l1-weight", type=float, default=0.0)
    parser.add_argument("--b-edge-l1-weight", type=float, default=0.1)
    parser.add_argument("--job-dir", type=Path, default=Path("outputs/experiments/edge-l1-ab-002-ssim-posthoc"))
    parser.add_argument("--previous-job", type=Path, help="Paused pre-SSIM-revision job; preserve its provenance")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume-job", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    a_path, b_path = project_path(ROOT, args.a_config), project_path(ROOT, args.b_config)
    a, b = load_yaml_config(a_path), load_yaml_config(b_path)
    verify_pair(a, b, a_weight=args.a_edge_l1_weight, b_weight=args.b_edge_l1_weight)
    job = project_path(ROOT, args.job_dir)
    status_path = job / "status.json"
    if job.exists() and not args.resume_job:
        raise ValueError("Job directory exists; use --resume-job after checking its status")
    if args.resume_job and status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if process_alive(old.get("runner_pid")) or process_alive(old.get("child_pid")):
            raise ValueError("An existing runner or training child is still active")
    job.mkdir(parents=True, exist_ok=True)
    status = {"started_at": now(), "runner_pid": os.getpid(), "child_pid": None,
              "phase": "preflight", "target_epochs_per_run": args.epochs,
              "A": {"run": args.a_run, "state": "pending", "edge_l1_weight": args.a_edge_l1_weight},
              "B": {"run": args.b_run, "state": "queued", "edge_l1_weight": args.b_edge_l1_weight},
              "test_accessed": False, "commands": []}
    write_json(status_path, status)
    sleep_guard = IdleSleepGuard()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the declared formal experiments")
        status["cuda_device"] = torch.cuda.get_device_name(0)
        status["ssim_schedule"] = ssim_schedule_from_config(a)
        status["idle_sleep_inhibited"] = sleep_guard.start()
        source_paths = [ROOT / name for name in (
            "src/trainer.py", "src/losses.py", "src/edge_validation.py", "src/metrics.py", "src/dataset.py",
            "src/utils.py", "src/post_selection.py", "scripts/03_finetune.py", "scripts/09_plot_selection_comparison.py",
            "scripts/10_run_edge_l1_ab.py", "scripts/11_compare_edge_l1_ab.py", "scripts/12_edge_l1_status.py")]
        source_paths += [a_path, b_path, project_path(ROOT, a["model"]["pretrained_checkpoint"])]
        sources = {str(path): digest(path) for path in source_paths}
        inputs = image_hashes(a)
        if args.previous_job and not args.resume_job:
            previous = project_path(ROOT, args.previous_job)
            old_status = json.loads((previous / "status.json").read_text(encoding="utf-8"))
            if old_status["phase"] != "paused" or process_alive(old_status.get("runner_pid")) or process_alive(old_status.get("child_pid")):
                raise ValueError("Previous job must be paused with no active runner or child")
            if old_status["A"]["run"] != args.a_run or old_status["B"]["run"] != args.b_run:
                raise ValueError("Previous job has different experiment identities")
            old_manifest = json.loads((previous / "frozen_manifest.json").read_text(encoding="utf-8"))
            if old_manifest["train_validation_images"] != inputs:
                raise ValueError("Train/validation images changed since the paused job")
            allowed_changes = {str(ROOT / name) for name in ("src/trainer.py", "src/metrics.py")}
            allowed_changes.update((str(a_path), str(b_path)))
            for path, old_hash in old_manifest["sources"].items():
                if path not in allowed_changes and digest(path) != old_hash:
                    raise ValueError(f"Unrelated frozen training source changed: {path}")
            for path, config in ((a_path, a), (b_path, b)):
                verify_ssim_revision(load_yaml_config(previous / "source" / path.relative_to(ROOT)), config)
            migration = job / "migration"
            migration.mkdir()
            source_last = ROOT / "checkpoints/finetuned" / args.a_run / "last.pth"
            pause = json.loads((previous / "pause_record_20261026.json").read_text(encoding="utf-8"))
            if digest(source_last) != pause["A"]["checkpoint_sha256"]:
                raise ValueError("Paused A checkpoint changed before the declared reporting revision")
            shutil.copyfile(source_last, migration / "previous_last.pth")
            shutil.copyfile(ROOT / "outputs/metrics" / args.a_run / "metrics.csv", migration / "previous_metrics.csv")
            shutil.copyfile(previous / "frozen_manifest.json", migration / "previous_frozen_manifest.json")
            revision = {"authorized_change": "User requested SSIM after selection and resumed training",
                        "previous_job": str(previous), "from_epoch": pause["A"]["completed_epochs"],
                        "ssim_schedule": "post_selection", "selection_and_loss_unchanged": True,
                        "checkpoint_sha256": pause["A"]["checkpoint_sha256"], "created_at": now()}
            write_json(migration / "revision.json", revision)
            status["revision"] = revision
            old_status.update(phase="superseded", superseded_by=str(job), updated_at=now())
            write_json(previous / "status.json", old_status)
        manifest_path = job / "frozen_manifest.json"
        manifest = {"sources": sources, "train_validation_images": inputs}
        if args.resume_job and manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Source/config/data changed since this job was declared")
        else:
            write_json(manifest_path, manifest)
            for path in source_paths:
                if path.suffix not in {".py", ".yaml"}:
                    continue
                relative = path.relative_to(ROOT)
                target = job / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for label, run, config_path, config in (("A", args.a_run, a_path, a), ("B", args.b_run, b_path, b)):
            if {str(path): digest(path) for path in source_paths} != sources or image_hashes(config) != inputs:
                raise ValueError("Training source, initialization or data changed between A and B")
            completed = inspect_run(run, config, args.epochs)
            status[label].update(progress(run))
            if completed < args.epochs:
                command = [sys.executable, "-u", str(ROOT / "scripts/03_finetune.py"), "--config", str(config_path),
                           "--epochs", str(args.epochs), "--device", "cuda"]
                if completed:
                    command += ["--resume", str(ROOT / "checkpoints/finetuned" / run / "last.pth")]
                else:
                    command += ["--experiment-name", run]
                status["commands"].append(command)
                status["phase"] = f"training_{label}"
                status[label]["state"] = "running"
                with (job / f"train_{label}.stdout.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    status["child_pid"] = child.pid
                    write_json(status_path, status)
                    while child.poll() is None:
                        status[label].update(progress(run))
                        status["updated_at"] = now()
                        write_json(status_path, status)
                        time.sleep(10)
                status["child_pid"] = None
                if child.returncode:
                    raise RuntimeError(f"{label} training failed with exit code {child.returncode}; see train_{label}.stdout.log")
                if inspect_run(run, config, args.epochs) != args.epochs:
                    raise RuntimeError(f"{label} did not complete the declared training budget")
            status[label].update(progress(run))
            status[label]["state"] = "complete"
            write_json(status_path, status)
        status["phase"] = "analysis"
        write_json(status_path, status)
        command = [sys.executable, str(ROOT / "scripts/11_compare_edge_l1_ab.py"),
                   "--a-run", args.a_run, "--b-run", args.b_run, "--epochs", str(args.epochs),
                   "--a-edge-l1-weight", str(args.a_edge_l1_weight), "--b-edge-l1-weight", str(args.b_edge_l1_weight),
                   "--output-dir", str(job / "analysis")]
        with (job / "analysis.stdout.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        status.update(phase="complete", finished_at=now(), analysis_path=str(job / "analysis"))
        write_json(status_path, status)
    except BaseException as exc:
        status.update(phase="failed", error=str(exc), failed_at=now())
        write_json(status_path, status)
        raise
    finally:
        sleep_guard.close()


if __name__ == "__main__":
    main()
