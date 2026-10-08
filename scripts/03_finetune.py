"""Formal QuickSRNet-Small 2x fine-tuning entry point.

Examples (run from the repository root):
    python .\scripts\03_finetune.py --config configs/base.yaml
    python .\scripts\03_finetune.py --config configs/base.yaml --dryrun
    python .\scripts\03_finetune.py --config configs/base.yaml --dryrun \
        --resume checkpoints/finetuned/<run>/last.pth --epochs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PairedImageDataset  # noqa: E402
from src.trainer import (  # noqa: E402
    Trainer,
    build_optimizer,
    build_quicksrnet_small_from_local_checkpoint,
    read_training_checkpoint_metadata,
    resolve_device,
    verify_checkpoint_reload,
)
from src.utils import (  # noqa: E402
    create_logger,
    ensure_dryrun_pair,
    load_yaml_config,
    make_run_name,
    project_path,
    require_mapping,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune QuickSRNet-Small 2x on paired images")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use data/dryrun and training-pair self-check validation",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved last.pth")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epoch count")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--experiment-name", default=None, help="Optional output subdirectory name")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _select_run_name(
    args: argparse.Namespace, resume_path: Path | None, dryrun: bool
) -> str:
    candidate: str | None = None
    if args.experiment_name:
        candidate = args.experiment_name
    elif resume_path is not None:
        metadata = read_training_checkpoint_metadata(resume_path)
        run_name = metadata.get("run_name")
        if isinstance(run_name, str) and run_name:
            candidate = run_name
    if candidate is None:
        candidate = make_run_name("dryrun" if dryrun else "gaming")
    if candidate in {".", ".."} or Path(candidate).name != candidate:
        raise ValueError(
            f"Experiment name must be one directory name without path separators, got {candidate!r}"
        )
    return candidate


def run(args: argparse.Namespace) -> None:
    config_path = project_path(ROOT, args.config)
    config = load_yaml_config(config_path)
    model_cfg = require_mapping(config, "model")
    data_cfg = require_mapping(config, "data")
    training_cfg = require_mapping(config, "training")
    validation_cfg = require_mapping(config, "validation")
    output_cfg = require_mapping(config, "output")

    model_name = str(model_cfg.get("name", ""))
    if model_name.casefold() != "quicksrnet_small":
        raise ValueError(f"Only model.name=quicksrnet_small is supported, got '{model_name}'")
    scale_factor = _positive_int(model_cfg.get("scale_factor", 2), "model.scale_factor")
    data_scale = _positive_int(data_cfg.get("scale_factor", scale_factor), "data.scale_factor")
    if data_scale != scale_factor:
        raise ValueError(
            f"model.scale_factor ({scale_factor}) and data.scale_factor ({data_scale}) disagree"
        )
    project_cfg = config.get("project", {})
    if isinstance(project_cfg, dict) and "scale_factor" in project_cfg:
        project_scale = _positive_int(project_cfg["scale_factor"], "project.scale_factor")
        if project_scale != scale_factor:
            raise ValueError(
                f"project.scale_factor ({project_scale}) and model.scale_factor "
                f"({scale_factor}) disagree"
            )
    if list(data_cfg.get("rgb_range", [0.0, 1.0])) != [0.0, 1.0]:
        raise ValueError("data.rgb_range must be [0.0, 1.0] for QuickSRNet")
    if str(training_cfg.get("loss", "l1")).casefold() != "l1":
        raise ValueError("Only training.loss=l1 is currently supported")
    if str(validation_cfg.get("metric", "psnr")).casefold() != "psnr":
        raise ValueError("Only validation.metric=psnr is currently supported")

    seed = _positive_int(config.get("seed", 42), "seed", allow_zero=True)
    seed_everything(seed)
    requested_device = args.device or str(config.get("device", "auto"))
    device = resolve_device(requested_device)
    pretrained = project_path(ROOT, model_cfg["pretrained_checkpoint"])

    resume_path = project_path(ROOT, args.resume) if args.resume is not None else None
    run_name = _select_run_name(args, resume_path, args.dryrun)

    if args.dryrun:
        dryrun_cfg = require_mapping(config, "dryrun")
        train_lr_dir = project_path(ROOT, dryrun_cfg.get("lr", "data/dryrun/LR"))
        train_hr_dir = project_path(ROOT, dryrun_cfg.get("hr", "data/dryrun/HR"))
        created = ensure_dryrun_pair(train_lr_dir, train_hr_dir, scale_factor)
        if created:
            print(f"Created deterministic synthetic dry-run pair under {train_lr_dir.parent}")
        val_lr_dir, val_hr_dir = train_lr_dir, train_hr_dir
        validation_is_independent = False
        epochs = _positive_int(
            args.epochs if args.epochs is not None else dryrun_cfg.get("epochs", 3),
            "dryrun.epochs",
        )
        batch_size = _positive_int(dryrun_cfg.get("batch_size", 1), "dryrun.batch_size")
        lr_patch_size = _positive_int(
            dryrun_cfg.get("lr_patch_size", 64), "dryrun.lr_patch_size"
        )
        augment = bool(dryrun_cfg.get("augment", True))
    else:
        train_lr_dir = project_path(ROOT, data_cfg["train_lr"])
        train_hr_dir = project_path(ROOT, data_cfg["train_hr"])
        val_lr_dir = project_path(ROOT, data_cfg["val_lr"])
        val_hr_dir = project_path(ROOT, data_cfg["val_hr"])
        validation_is_independent = True
        epochs = _positive_int(
            args.epochs if args.epochs is not None else training_cfg.get("epochs", 50),
            "training.epochs",
        )
        batch_size = _positive_int(training_cfg.get("batch_size", 8), "training.batch_size")
        lr_patch_size = _positive_int(data_cfg.get("lr_patch_size", 64), "data.lr_patch_size")
        augment = bool(data_cfg.get("augment", True))

    # Dataset construction performs eager image decode and strict pair/scale checks.
    train_dataset = PairedImageDataset(
        train_lr_dir,
        train_hr_dir,
        scale_factor=scale_factor,
        lr_patch_size=lr_patch_size,
        augment=augment,
    )
    val_dataset = PairedImageDataset(
        val_lr_dir,
        val_hr_dir,
        scale_factor=scale_factor,
        lr_patch_size=None,
        augment=False,
    )

    num_workers = _positive_int(data_cfg.get("num_workers", 0), "data.num_workers", allow_zero=True)
    val_batch_size = _positive_int(validation_cfg.get("batch_size", 1), "validation.batch_size")
    pin_memory = bool(data_cfg.get("pin_memory", True)) and device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    checkpoint_dir = project_path(ROOT, output_cfg["checkpoint_dir"]) / run_name
    log_dir = project_path(ROOT, output_cfg["log_dir"]) / run_name
    metric_dir = project_path(ROOT, output_cfg["metric_dir"]) / run_name
    sample_dir = project_path(ROOT, output_cfg["sample_dir"]) / run_name
    logger = create_logger(f"quicksrnet.{run_name}", log_dir / "train.log")
    logger.info("Configuration: %s", config_path)
    logger.info("Run: %s | mode: %s | device: %s", run_name, "dry-run" if args.dryrun else "formal", device)
    logger.info("Train pairs: %d | validation pairs: %d", len(train_dataset), len(val_dataset))
    logger.info("Train LR/HR: %s | %s", train_lr_dir, train_hr_dir)
    logger.info("Validation LR/HR: %s | %s", val_lr_dir, val_hr_dir)
    if not validation_is_independent:
        logger.warning(
            "Dry-run validation reuses the training pair(s). Metrics are pipeline self-checks, "
            "not independent validation or evidence of generalization."
        )

    model = build_quicksrnet_small_from_local_checkpoint(pretrained, scale_factor=scale_factor)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info(
        "Loaded local pretrained model: %s | parameters: %d | trainable: %d",
        pretrained,
        parameter_count,
        trainable_count,
    )
    optimizer = build_optimizer(
        model,
        str(training_cfg.get("optimizer", "adam")),
        float(training_cfg.get("learning_rate", 1e-4)),
    )
    save_best = bool(validation_cfg.get("save_best", True))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        scale_factor=scale_factor,
        config=config,
        run_name=run_name,
        pretrained_checkpoint=pretrained,
        checkpoint_dir=checkpoint_dir,
        metric_dir=metric_dir,
        sample_dir=sample_dir,
        logger=logger,
        validation_is_independent=validation_is_independent,
        save_best=save_best,
        amp=bool(training_cfg.get("amp", False)),
        save_samples=bool(validation_cfg.get("save_samples", True)),
    )
    start_epoch = trainer.resume(resume_path) if resume_path is not None else 0
    last_checkpoint = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

    verification_batch = next(iter(val_loader))
    reload_difference = verify_checkpoint_reload(
        checkpoint_path=last_checkpoint,
        pretrained_checkpoint=pretrained,
        scale_factor=scale_factor,
        trained_model=trainer.model,
        sample_lr=verification_batch["lr"][:1],
        device=device,
    )
    logger.info("Checkpoint reload inference verified (max abs output diff %.3g)", reload_difference)
    logger.info("Training complete. Last checkpoint: %s", last_checkpoint)
    if validation_is_independent and save_best:
        logger.info("Best validation checkpoint: %s", checkpoint_dir / "best.pth")
    elif not validation_is_independent and save_best:
        logger.warning(
            "Dry run complete: best_self_check.pth is deliberately not named best.pth because "
            "the validation data is not independent."
        )


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError, RuntimeError, AssertionError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

