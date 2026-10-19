"""Model loading, training, validation, checkpointing, and resume support."""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from qai_hub_models.models.quicksrnetsmall.model import (
    QuickSRNetSmall,
    _load_quicksrnet_source_model,
)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be one of auto/cpu/cuda, got '{requested}'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_torch_file(path: Path, *, trusted: bool) -> Any:
    """Use safe tensor-only loading where possible; official QAI files need pickle."""

    try:
        return torch.load(path, map_location="cpu", weights_only=not trusted)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location="cpu")


def build_quicksrnet_small_from_local_checkpoint(
    checkpoint_path: str | Path,
    *,
    scale_factor: int = 2,
) -> QuickSRNetSmall:
    """Build the official Small architecture without downloading any asset.

    This intentionally mirrors the loading path already verified by
    ``02_train_sanity_check.py``. The checkpoint must be the trusted official
    Qualcomm file and its backbone weights must live under ``state_dict``.
    """

    path = Path(checkpoint_path).resolve()
    if scale_factor != 2:
        raise ValueError(
            f"This pipeline is intentionally limited to QuickSRNet-Small 2x, got {scale_factor}x"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Official pretrained checkpoint not found: {path}. Copy the Qualcomm "
            "QuickSRNet-Small 2x .pth.tar file into checkpoints/pretrained/."
        )

    backbone = _load_quicksrnet_source_model(scale_factor, 32, 2, False)
    checkpoint = _load_torch_file(path, trusted=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Expected an official QAI checkpoint dictionary containing 'state_dict': {path}"
        )
    try:
        backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint weights are incompatible with QuickSRNet-Small 2x: {path}\n{exc}"
        ) from exc
    model = QuickSRNetSmall(backbone, scale_factor)
    model.requires_grad_(True)
    return model


def build_optimizer(model: torch.nn.Module, name: str, learning_rate: float) -> torch.optim.Optimizer:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive and finite, got {learning_rate}")
    if name.casefold() != "adam":
        raise ValueError(f"Only the Adam optimizer is currently supported, got '{name}'")
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _backbone(model: torch.nn.Module) -> torch.nn.Module:
    inner = getattr(model, "model", None)
    if not isinstance(inner, torch.nn.Module):
        raise TypeError("Expected the official QuickSRNetSmall wrapper with a '.model' backbone")
    return inner


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def read_training_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = _load_torch_file(checkpoint_path, trusted=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Training checkpoint is not a dictionary: {checkpoint_path}")
    return {
        "run_name": checkpoint.get("run_name"),
        "epoch": checkpoint.get("epoch"),
        "scale_factor": checkpoint.get("scale_factor"),
    }


class Trainer:
    """A compact native-PyTorch trainer for full-model fine-tuning."""

    CSV_FIELDS = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_psnr_db",
        "learning_rate",
        "epoch_seconds",
        "validation_scope",
    )

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        scale_factor: int,
        config: dict[str, Any],
        run_name: str,
        pretrained_checkpoint: Path,
        checkpoint_dir: Path,
        metric_dir: Path,
        sample_dir: Path,
        logger: logging.Logger,
        validation_is_independent: bool,
        save_best: bool = True,
        amp: bool = False,
        save_samples: bool = True,
    ) -> None:
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scale_factor = scale_factor
        self.config = config
        self.run_name = run_name
        self.pretrained_checkpoint = pretrained_checkpoint.resolve()
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.metric_dir = metric_dir.resolve()
        self.sample_dir = sample_dir.resolve()
        self.logger = logger
        self.validation_is_independent = validation_is_independent
        self.save_best = save_best
        self.save_samples = save_samples
        self.best_val_psnr = -math.inf
        self.history: list[dict[str, Any]] = []
        self.amp_enabled = bool(amp and device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        for directory in (self.checkpoint_dir, self.metric_dir, self.sample_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def resume(self, checkpoint_path: str | Path) -> int:
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        checkpoint = _load_torch_file(path, trusted=False)
        required = {"state_dict", "optimizer_state_dict", "epoch"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            missing = sorted(required - set(checkpoint if isinstance(checkpoint, dict) else {}))
            raise ValueError(f"Invalid training checkpoint {path}; missing keys: {missing}")
        checkpoint_scale = int(checkpoint.get("scale_factor", self.scale_factor))
        if checkpoint_scale != self.scale_factor:
            raise ValueError(
                f"Resume checkpoint is {checkpoint_scale}x, but configuration is "
                f"{self.scale_factor}x: {path}"
            )
        try:
            _backbone(self.model).load_state_dict(checkpoint["state_dict"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Could not restore model/optimizer state from {path}: {exc}") from exc
        _move_optimizer_state(self.optimizer, self.device)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state and self.amp_enabled:
            self.scaler.load_state_dict(scaler_state)
        self.best_val_psnr = float(checkpoint.get("best_val_psnr", -math.inf))
        history = checkpoint.get("metrics", [])
        if isinstance(history, list):
            self.history = history
        epoch = int(checkpoint["epoch"])
        if epoch < 0:
            raise ValueError(f"Resume checkpoint has invalid epoch {epoch}: {path}")
        self.logger.info("Resumed model, optimizer, and epoch %d from %s", epoch, path)
        self.logger.info(
            "Resume state | Adam parameter states: %d | learning rates: %s | "
            "history rows: %d | global best validation PSNR: %.6f dB | AMP scale: %.0f",
            len(self.optimizer.state),
            [group["lr"] for group in self.optimizer.param_groups],
            len(self.history),
            self.best_val_psnr,
            self.scaler.get_scale(),
        )
        return epoch

    def _check_output(self, output: torch.Tensor, target: torch.Tensor, context: str) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Model returned {type(output).__name__}, expected Tensor ({context})")
        if output.shape != target.shape:
            raise RuntimeError(
                f"Model output shape {tuple(output.shape)} does not match HR shape "
                f"{tuple(target.shape)} ({context})"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Model output contains NaN/Inf ({context})")

    def train_one_epoch(self, loader: DataLoader[Any], epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_images = 0
        for batch_index, batch in enumerate(loader, start=1):
            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(lr)
                self._check_output(output, hr, f"epoch {epoch}, batch {batch_index}")
                loss = torch.nn.functional.l1_loss(output, hr)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_index}: "
                    f"{loss.detach().item()}"
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("No gradients were produced; model parameters may be frozen")
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError(f"Gradient contains NaN/Inf at epoch {epoch}, batch {batch_index}")
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = lr.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_images += batch_size
            if batch_index == 1 or batch_index % 20 == 0:
                self.logger.info(
                    "Epoch %d training | batch %d | images %d | mean L1 %.6f",
                    epoch, batch_index, total_images, total_loss / total_images,
                )

        if total_images == 0:
            raise RuntimeError("Training DataLoader produced no batches")
        return total_loss / total_images

    def validate(self, loader: DataLoader[Any], epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_l1 = 0.0
        total_psnr = 0.0
        total_images = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                lr = batch["lr"].to(self.device, non_blocking=True)
                hr = batch["hr"].to(self.device, non_blocking=True)
                output = self.model(lr)
                self._check_output(output, hr, f"validation epoch {epoch}, batch {batch_index}")
                per_image_l1 = torch.mean(torch.abs(output - hr), dim=(1, 2, 3))
                per_image_mse = torch.mean((output.clamp(0.0, 1.0) - hr) ** 2, dim=(1, 2, 3))
                per_image_psnr = torch.where(
                    per_image_mse == 0,
                    torch.full_like(per_image_mse, float("inf")),
                    -10.0 * torch.log10(per_image_mse),
                )
                total_l1 += per_image_l1.sum().item()
                total_psnr += per_image_psnr.sum().item()
                total_images += lr.shape[0]
                if batch_index == 1 or batch_index % 60 == 0:
                    self.logger.info(
                        "Epoch %d validation | images %d | mean PSNR %.3f dB",
                        epoch, total_images, total_psnr / total_images,
                    )
        if total_images == 0:
            raise RuntimeError("Validation DataLoader produced no batches")
        return total_l1 / total_images, total_psnr / total_images

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "state_dict": _backbone(self.model).state_dict(),
            "state_dict_format": "quicksrnet_backbone",
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "config": self.config,
            "best_val_psnr": self.best_val_psnr,
            "metrics": self.history,
            "scale_factor": self.scale_factor,
            "run_name": self.run_name,
            "validation_is_independent": self.validation_is_independent,
            "source_pretrained": str(self.pretrained_checkpoint),
        }

    def save_checkpoint(self, filename: str, epoch: int) -> Path:
        path = self.checkpoint_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary)
        temporary.replace(path)
        return path

    def _write_metrics(self) -> Path:
        path = self.metric_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def _save_validation_sample(self, loader: DataLoader[Any]) -> None:
        batch = next(iter(loader))
        lr = batch["lr"][:1].to(self.device)
        hr = batch["hr"][:1]
        self.model.eval()
        with torch.inference_mode():
            output = self.model(lr).clamp(0.0, 1.0).cpu()

        def tensor_image(tensor: torch.Tensor) -> Image.Image:
            array = (
                tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            return Image.fromarray(array)

        lr_image = tensor_image(lr.cpu())
        hr_image = tensor_image(hr)
        sr_image = tensor_image(output)
        resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        panels = [lr_image.resize(hr_image.size, resampling), sr_image, hr_image]
        labels = ["LR (bicubic display)", "QuickSRNet SR", "Ground-truth HR"]
        canvas = Image.new("RGB", (hr_image.width * 3, hr_image.height + 34), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, labels)):
            left = index * hr_image.width
            canvas.paste(panel, (left, 34))
            draw.text((left + 8, 10), label, fill=(20, 25, 30))
        canvas.save(self.sample_dir / "latest_validation_sample.png")

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        start_epoch: int = 0,
    ) -> Path:
        if start_epoch >= self.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch}, but configured total "
                f"epochs is {self.epochs}. Increase training.epochs or pass --epochs."
            )

        scope = "independent_validation" if self.validation_is_independent else "training_pair_self_check"
        self.logger.info(
            "Starting epochs %d..%d on %s (AMP=%s, validation=%s)",
            start_epoch + 1,
            self.epochs,
            self.device,
            self.amp_enabled,
            scope,
        )
        last_path: Path | None = None
        for epoch in range(start_epoch + 1, self.epochs + 1):
            start_time = time.perf_counter()
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_psnr = self.validate(val_loader, epoch)
            elapsed = time.perf_counter() - start_time
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr_db": val_psnr,
                "learning_rate": learning_rate,
                "epoch_seconds": elapsed,
                "validation_scope": scope,
            }
            self.history.append(row)

            improved = val_psnr > self.best_val_psnr
            if improved:
                self.best_val_psnr = val_psnr
            last_path = self.save_checkpoint("last.pth", epoch)
            if improved and self.save_best:
                best_name = "best.pth" if self.validation_is_independent else "best_self_check.pth"
                best_path = self.save_checkpoint(best_name, epoch)
                self.logger.info("Saved improved %s checkpoint: %s", scope, best_path)
            metrics_path = self._write_metrics()
            if self.save_samples:
                self._save_validation_sample(val_loader)
            self.logger.info(
                "Epoch %d/%d | train L1 %.6f | val L1 %.6f | val PSNR %.3f dB | "
                "lr %.3g | %.2fs",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_psnr,
                learning_rate,
                elapsed,
            )
            self.logger.info("Saved last checkpoint: %s | metrics: %s", last_path, metrics_path)

        assert last_path is not None
        return last_path


def verify_checkpoint_reload(
    *,
    checkpoint_path: str | Path,
    pretrained_checkpoint: str | Path,
    scale_factor: int,
    trained_model: torch.nn.Module,
    sample_lr: torch.Tensor,
    device: torch.device,
) -> float:
    """Reload a saved training checkpoint and compare one inference output."""

    path = Path(checkpoint_path).resolve()
    checkpoint = _load_torch_file(path, trusted=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Training checkpoint has no 'state_dict': {path}")
    reloaded = build_quicksrnet_small_from_local_checkpoint(
        pretrained_checkpoint, scale_factor=scale_factor
    ).to(device)
    _backbone(reloaded).load_state_dict(checkpoint["state_dict"], strict=True)
    trained_model.eval()
    reloaded.eval()
    sample_lr = sample_lr.to(device)
    with torch.inference_mode():
        expected = trained_model(sample_lr)
        actual = reloaded(sample_lr)
    difference = (expected - actual).abs().max().item()
    if difference > 1e-6:
        raise AssertionError(f"Checkpoint reload output mismatch: max abs diff={difference}")
    return difference

