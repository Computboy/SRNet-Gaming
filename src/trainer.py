"""Model loading, training, validation, checkpointing, and resume support."""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.metrics import rgb_psnr_per_image, rgb_ssim, ssim_schedule_from_config, selection_validation_settings
from src.edge_validation import (
    EdgeRegionMetric, EdgeValidationProtocol, json_safe, select_epochs, weighted_score,
)
from src.losses import EdgeAwareL1, TrainingLossProtocol

from qai_hub_models.models.quicksrnetsmall.model import (
    QuickSRNetSmall,
    _load_quicksrnet_source_model,
)
from qai_hub_models.models.quicksrnetmedium.model import QuickSRNetMedium


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
    """Backward-compatible Small factory used by the completed A/B tools."""
    return build_quicksrnet_from_local_checkpoint(checkpoint_path, model_name="quicksrnet_small", scale_factor=scale_factor)


MODEL_VARIANTS = {
    "quicksrnet_small": (QuickSRNetSmall, 32, 2),
    "quicksrnet_medium": (QuickSRNetMedium, 32, 5),
}


def build_quicksrnet_from_local_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_name: str = "quicksrnet_small",
    scale_factor: int = 2,
) -> torch.nn.Module:
    """Build an official Small/Medium architecture and strictly load local weights.

    This intentionally mirrors the loading path already verified by
    ``02_train_sanity_check.py``. The checkpoint must be the trusted official
    Qualcomm file and its backbone weights must live under ``state_dict``.
    """

    path = Path(checkpoint_path).resolve()
    model_name = model_name.casefold()
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported QuickSRNet variant: {model_name}")
    wrapper, channels, intermediate_layers = MODEL_VARIANTS[model_name]
    if scale_factor != 2:
        raise ValueError(
            f"This pipeline is intentionally limited to QuickSRNet 2x, got {scale_factor}x"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Official pretrained checkpoint not found: {path}. Copy the Qualcomm "
            f"{model_name} 2x .pth.tar file into checkpoints/pretrained/."
        )

    backbone = _load_quicksrnet_source_model(scale_factor, channels, intermediate_layers, False)
    checkpoint = _load_torch_file(path, trusted=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Expected an official QAI checkpoint dictionary containing 'state_dict': {path}"
        )
    try:
        backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint weights are incompatible with {model_name} 2x: {path}\n{exc}"
        ) from exc
    model = wrapper(backbone, scale_factor)
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
        raise TypeError("Expected an official QuickSRNet wrapper with a '.model' backbone")
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
    EDGE_CSV_FIELDS = (
        "val_ssim", "val_edge_region_psnr_db", "val_edge_images",
        "val_edge_coverage", "val_weighted_score", "resumed_from_epoch", "val_ssim_status",
    )
    LOSS_CSV_FIELDS = (
        "train_global_l1", "train_edge_l1", "train_total_loss",
        "train_edge_coverage", "train_edge_empty_masks",
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
        validation_fingerprint: str | None = None,
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
        self.edge_protocol = EdgeValidationProtocol.from_config(config)
        self.ssim_schedule = ssim_schedule_from_config(config)
        self.logger.info("SSIM schedule: %s (does not participate in checkpoint selection)", self.ssim_schedule)
        if self.edge_protocol.enabled and not self.save_best:
            raise ValueError("Edge selection comparison requires validation.save_best=true")
        self.edge_metric = EdgeRegionMetric(self.edge_protocol).to(device) if self.edge_protocol.enabled else None
        self.loss_protocol = TrainingLossProtocol.from_config(config)
        loss_mask = self.edge_metric
        if loss_mask is None and self.loss_protocol.mode == "l1_edge":
            loss_mask = EdgeRegionMetric(self.edge_protocol).to(device)
        self.loss_function = EdgeAwareL1(self.loss_protocol, loss_mask).to(device)
        self.record_loss_components = self.edge_protocol.enabled or self.loss_protocol.mode == "l1_edge"
        self.last_train_components: dict[str, Any] = {}
        self.last_validation_details: dict[str, Any] = {}
        self.validation_fingerprint = validation_fingerprint
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
        saved_model_name = checkpoint.get("config", {}).get("model", {}).get("name", "quicksrnet_small")
        if saved_model_name != self.config.get("model", {}).get("name", "quicksrnet_small"):
            raise ValueError("Cannot resume a checkpoint from a different QuickSRNet model variant")
        if checkpoint_scale != self.scale_factor:
            raise ValueError(
                f"Resume checkpoint is {checkpoint_scale}x, but configuration is "
                f"{self.scale_factor}x: {path}"
            )
        expected_protocol = self.edge_protocol.definition() if self.edge_protocol.enabled else None
        if checkpoint.get("edge_validation_protocol") != expected_protocol:
            raise ValueError(
                "Cannot change the edge validation/selection protocol during resume. "
                "Start a new run from the official pretrained weights instead."
            )
        if self.edge_protocol.enabled and checkpoint.get("validation_fingerprint") != self.validation_fingerprint:
            raise ValueError("Validation image contents changed since checkpoint creation; start a new run")
        if selection_validation_settings(checkpoint.get("config", {})) != selection_validation_settings(self.config):
            raise ValueError("Cannot change validation settings other than the SSIM reporting schedule during resume")
        old_schedule = ssim_schedule_from_config(checkpoint.get("config", {}))
        if old_schedule != self.ssim_schedule:
            self.logger.info("SSIM reporting schedule revision on resume: %s -> %s", old_schedule, self.ssim_schedule)
        saved_loss = checkpoint.get("training_loss_protocol")
        if saved_loss is None:
            # Legacy checkpoints used global L1 only. Derive its explicit protocol.
            old_config = checkpoint.get("config", {})
            saved_loss = TrainingLossProtocol.from_config(old_config).definition(
                EdgeValidationProtocol.from_config(old_config)
            )
        if saved_loss != self.loss_protocol.definition(self.edge_protocol):
            raise ValueError("Cannot change training Loss mode, edge weight or mask protocol during resume")
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
            if self.edge_protocol.enabled:
                for row in self.history:
                    row.setdefault("val_ssim_status", "per_epoch" if row.get("val_ssim") is not None else "deferred_post_selection")
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
        component_sums = {"global_l1": 0.0, "edge_l1": 0.0, "edge_coverage": 0.0}
        empty_masks = 0
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
                components = self.loss_function(output, hr)
                loss = components["total_loss"]
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
                if not self.amp_enabled:
                    raise RuntimeError(f"Gradient contains NaN/Inf at epoch {epoch}, batch {batch_index}")
                self.logger.warning(
                    "AMP gradient overflow at epoch %d, batch %d: GradScaler will skip the update and lower the loss scale",
                    epoch,
                    batch_index,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = lr.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_images += batch_size
            if self.record_loss_components:
                for name in component_sums:
                    component_sums[name] += components[name].detach().item() * batch_size
                empty_masks += components["empty_masks"].item()
            if batch_index == 1 or batch_index % 20 == 0:
                self.logger.info(
                    "Epoch %d training | batch %d | images %d | mean total loss %.6f",
                    epoch, batch_index, total_images, total_loss / total_images,
                )

        if total_images == 0:
            raise RuntimeError("Training DataLoader produced no batches")
        if self.record_loss_components:
            self.last_train_components = {
                "train_global_l1": component_sums["global_l1"] / total_images,
                "train_edge_l1": component_sums["edge_l1"] / total_images,
                "train_total_loss": total_loss / total_images,
                "train_edge_coverage": component_sums["edge_coverage"] / total_images,
                "train_edge_empty_masks": empty_masks,
            }
            self.logger.info("Training loss components: %s", self.last_train_components)
        return total_loss / total_images

    def validate(self, loader: DataLoader[Any], epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_l1 = 0.0
        total_psnr = 0.0
        total_images = 0
        total_ssim = total_edge_psnr = total_coverage = 0.0
        edge_images = 0
        per_image_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                lr = batch["lr"].to(self.device, non_blocking=True)
                hr = batch["hr"].to(self.device, non_blocking=True)
                output = self.model(lr)
                self._check_output(output, hr, f"validation epoch {epoch}, batch {batch_index}")
                per_image_l1 = torch.mean(torch.abs(output - hr), dim=(1, 2, 3))
                per_image_psnr = rgb_psnr_per_image(output, hr)
                total_l1 += per_image_l1.sum().item()
                total_psnr += per_image_psnr.sum().item()
                if self.edge_metric is not None:
                    edges = self.edge_metric(output, hr)
                    compute_ssim = self.ssim_schedule == "per_epoch"
                    if compute_ssim:
                        predictions = output.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                        references = hr.permute(0, 2, 3, 1).cpu().numpy()
                    for index in range(lr.shape[0]):
                        ssim = rgb_ssim(predictions[index], references[index]) if compute_ssim else None
                        valid = edges["pixels"][index].item() > 0
                        edge_psnr = edges["psnr_db"][index].item() if valid else None
                        coverage = edges["coverage"][index].item()
                        if ssim is not None:
                            total_ssim += ssim
                        total_coverage += coverage
                        if valid:
                            total_edge_psnr += edge_psnr
                            edge_images += 1
                        per_image_rows.append({
                            "key": batch["name"][index],
                            "psnr_db": per_image_psnr[index].item(), "ssim": ssim,
                            "edge_region_psnr_db": edge_psnr,
                            "edge_pixels": edges["pixels"][index].item(), "edge_coverage": coverage,
                        })
                total_images += lr.shape[0]
                if batch_index == 1 or batch_index % 60 == 0:
                    self.logger.info(
                        "Epoch %d validation | images %d | mean PSNR %.3f dB",
                        epoch, total_images, total_psnr / total_images,
                    )
        if total_images == 0:
            raise RuntimeError("Validation DataLoader produced no batches")
        if self.edge_metric is not None:
            mean_edge = total_edge_psnr / edge_images if edge_images else None
            self.last_validation_details = {
                "val_ssim": total_ssim / total_images if self.ssim_schedule == "per_epoch" else None,
                "val_ssim_status": "per_epoch" if self.ssim_schedule == "per_epoch" else "deferred_post_selection",
                "val_edge_region_psnr_db": mean_edge,
                "val_edge_images": edge_images,
                "val_edge_coverage": total_coverage / total_images,
                "val_weighted_score": weighted_score(
                    total_psnr / total_images, mean_edge, self.edge_protocol.weighted_alpha
                ),
            }
            validation_dir = self.metric_dir / "validation"
            validation_dir.mkdir(exist_ok=True)
            with (validation_dir / f"epoch{epoch:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
                writer.writeheader()
                writer.writerows(per_image_rows)
            self.logger.info("Additional validation metrics: %s", self.last_validation_details)
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
            "edge_validation_protocol": self.edge_protocol.definition() if self.edge_protocol.enabled else None,
            "validation_fingerprint": self.validation_fingerprint,
            "training_loss_protocol": self.loss_protocol.definition(self.edge_protocol),
            "ssim_schedule": self.ssim_schedule,
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
            fields = self.CSV_FIELDS + (self.EDGE_CSV_FIELDS if self.edge_protocol.enabled else ())
            if self.record_loss_components:
                fields += self.LOSS_CSV_FIELDS
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def _save_selection_comparison(self, epoch: int) -> None:
        """Retain every epoch so the global constrained selection remains exact."""
        epoch_dir = self.checkpoint_dir / "epochs"
        epoch_dir.mkdir(exist_ok=True)
        self.save_checkpoint(f"epochs/epoch{epoch:03d}.pth", epoch)
        choices = select_epochs(self.history, self.edge_protocol)
        previous_path = self.metric_dir / "selection_summary.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
        selected: dict[str, Any] = {}
        for strategy, row in choices.items():
            selected_epoch = int(row["epoch"])
            suffix = "" if self.validation_is_independent else "_self_check"
            filename = f"best_{strategy}{suffix}.pth"
            if (selected_epoch == epoch
                    or previous.get("selected", {}).get(strategy, {}).get("epoch") != selected_epoch
                    or not (self.checkpoint_dir / filename).exists()):
                source = epoch_dir / f"epoch{selected_epoch:03d}.pth"
                temporary = self.checkpoint_dir / (filename + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(self.checkpoint_dir / filename)
            selected[strategy] = {**row, "checkpoint": str(self.checkpoint_dir / filename)}
        best_psnr = max(float(row["val_psnr_db"]) for row in self.history)
        summary = {
            "completed_epoch": epoch,
            "validation_is_independent": self.validation_is_independent,
            "protocol": self.edge_protocol.definition(),
            "best_val_psnr_db": best_psnr,
            "constrained_psnr_floor_db": best_psnr - self.edge_protocol.psnr_tolerance_db,
            "selected": selected,
            "same_epoch_all_strategies": (
                len({row["epoch"] for row in selected.values()}) == 1 if len(selected) == 3 else None
            ),
            "test_set_used": False,
            "validation_fingerprint": self.validation_fingerprint,
            "training_loss_protocol": self.loss_protocol.definition(self.edge_protocol),
            "ssim_schedule": self.ssim_schedule,
        }
        temporary = previous_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(previous_path)

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
        if self.edge_protocol.enabled:
            missing = [row["epoch"] for row in self.history
                       if not (self.checkpoint_dir / "epochs" / f"epoch{int(row['epoch']):03d}.pth").is_file()]
            if missing:
                raise ValueError(f"Resume requires retained epoch checkpoints for exact selection; missing: {missing}")

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
            if self.edge_protocol.enabled:
                row.update(self.last_validation_details)
                row["resumed_from_epoch"] = start_epoch if epoch == start_epoch + 1 else 0
            if self.record_loss_components:
                row.update(self.last_train_components)
            self.history.append(row)

            improved = val_psnr > self.best_val_psnr
            if improved:
                self.best_val_psnr = val_psnr
            if self.edge_protocol.enabled and self.save_best:
                self._save_selection_comparison(epoch)
            last_path = self.save_checkpoint("last.pth", epoch)
            if improved and self.save_best:
                best_name = "best.pth" if self.validation_is_independent else "best_self_check.pth"
                best_path = self.save_checkpoint(best_name, epoch)
                self.logger.info("Saved improved %s checkpoint: %s", scope, best_path)
            metrics_path = self._write_metrics()
            if self.save_samples:
                self._save_validation_sample(val_loader)
            self.logger.info(
                "Epoch %d/%d | train total %.6f | val L1 %.6f | val PSNR %.3f dB | "
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
    reloaded = build_quicksrnet_from_local_checkpoint(
        pretrained_checkpoint, scale_factor=scale_factor,
        model_name=checkpoint.get("config", {}).get("model", {}).get("name", "quicksrnet_small"),
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

