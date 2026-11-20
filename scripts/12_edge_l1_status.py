"""Read A/B job progress without importing Torch or touching image data."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=ROOT / "outputs/experiments/edge-l1-ab-002-ssim-posthoc")
    args = parser.parse_args()
    status = json.loads((args.job_dir / "status.json").read_text(encoding="utf-8"))
    print(f"Phase: {status['phase']} | Updated: {status.get('updated_at', status['started_at'])}")
    print(f"GPU: {status.get('cuda_device')} | Runner PID: {status['runner_pid']} | Training PID: {status.get('child_pid')}")
    for label in ("A", "B"):
        run = status[label]
        print(f"{label}: {run['run']} | {run['state']} | {run.get('completed_epochs', 0)}/{status['target_epochs_per_run']} epochs")
        metrics = run.get("latest_metrics")
        if metrics:
            ssim = metrics.get("val_ssim") or "deferred until checkpoint selection"
            print(f"   PSNR={metrics['val_psnr_db']} dB | Edge-region PSNR={metrics['val_edge_region_psnr_db']} dB | SSIM={ssim}")
    if status.get("error"):
        print(f"ERROR: {status['error']}")
    if status.get("analysis_path"):
        print(f"Analysis: {status['analysis_path']}")


if __name__ == "__main__":
    main()
