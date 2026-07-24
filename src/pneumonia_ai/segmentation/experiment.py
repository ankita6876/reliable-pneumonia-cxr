"""Reproducible train/validation experiment runner for lung segmentation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import torch

from pneumonia_ai.segmentation.dataset import MontgomerySegmentationDataset
from pneumonia_ai.segmentation.losses import BCEWithLogitsDiceLoss
from pneumonia_ai.segmentation.model import UNet
from pneumonia_ai.segmentation.training import (
    build_dataloader,
    build_optimizer,
    build_scheduler,
    seed_everything,
    train_one_epoch,
    validate_one_epoch,
)
from pneumonia_ai.segmentation.transforms import EvaluationTransform, TrainingTransform


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one train/validation-only U-Net experiment."""

    train_csv: Path
    validation_csv: Path
    output_directory: Path
    image_size: int = 128
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    base_channels: int = 16
    depth: int = 3
    seed: int = 42
    patience: int = 5
    gradient_clip_norm: float | None = 1.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    threshold: float = 0.5
    num_workers: int = 0
    device: str = "cpu"
    resume: bool = False


def run_experiment(config: ExperimentConfig) -> list[dict[str, float | int]]:
    """Train with validation model selection, never loading a test split."""

    _validate_config(config)
    device = _resolve_device(config.device)
    seed_everything(config.seed)
    output_directory = config.output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    train_dataset = MontgomerySegmentationDataset(
        config.train_csv, image_size=config.image_size, transform=TrainingTransform()
    )
    validation_dataset = MontgomerySegmentationDataset(
        config.validation_csv,
        image_size=config.image_size,
        transform=EvaluationTransform(),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    model_config = _model_config(config)
    model = UNet(**model_config)
    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = build_scheduler(optimizer)
    loss_function = BCEWithLogitsDiceLoss(config.bce_weight, config.dice_weight)
    _write_json(output_directory / "resolved_config.json", _json_config(config))
    _write_json(
        output_directory / "run_manifest.json",
        _run_manifest(config, device, len(train_dataset), len(validation_dataset)),
    )

    start_epoch, best_dice, history, epochs_without_improvement = _resume_if_requested(
        config, output_directory, model, optimizer, scheduler, model_config
    )
    for epoch in range(start_epoch, config.epochs):
        started = time.monotonic()
        train_summary = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
            use_amp=device.type == "cuda",
        )
        validation_summary = validate_one_epoch(
            model, validation_loader, loss_function, device=device
        )
        scheduler.step()
        elapsed_seconds = time.monotonic() - started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_summary.loss,
            "validation_loss": validation_summary.loss,
            "train_dice": train_summary.dice,
            "validation_dice": validation_summary.dice,
            "validation_iou": validation_summary.iou,
            "validation_precision": validation_summary.precision,
            "validation_recall": validation_summary.sensitivity,
            "validation_specificity": validation_summary.specificity,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(row)
        is_best = validation_summary.dice > best_dice
        if is_best:
            best_dice = validation_summary.dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = _checkpoint_payload(
            model, optimizer, scheduler, epoch + 1, best_dice, model_config, config
        )
        _atomic_torch_save(checkpoint, output_directory / "latest_checkpoint.pt")
        if is_best:
            _atomic_torch_save(checkpoint, output_directory / "best_checkpoint.pt")
        _write_history(output_directory / "history.csv", history)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: train loss={train_summary.loss:.4f}, "
            f"validation loss={validation_summary.loss:.4f}, "
            f"validation Dice={validation_summary.dice:.4f}"
        )
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {epoch + 1} epochs (patience={config.patience}).")
            break
    return history


def _resume_if_requested(
    config: ExperimentConfig,
    output_directory: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: dict[str, int],
) -> tuple[int, float, list[dict[str, float | int]], int]:
    history_path = output_directory / "history.csv"
    if not config.resume:
        return 0, float("-inf"), [], 0
    checkpoint_path = output_directory / "latest_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, model_config, config.seed)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    history = _read_history(history_path)
    return int(checkpoint["epoch"]), float(checkpoint["best_validation_dice"]), history, 0


def _checkpoint_payload(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_validation_dice: float,
    model_config: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_validation_dice": best_validation_dice,
        "model_config": model_config,
        "training_config": _json_config(config),
        "seed": config.seed,
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any], model_config: dict[str, int], seed: int
) -> None:
    required = {"model_state", "optimizer_state", "epoch", "best_validation_dice", "model_config", "seed"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(sorted(missing))}")
    if checkpoint["model_config"] != model_config:
        raise ValueError("Checkpoint model configuration is incompatible with this run.")
    if checkpoint["seed"] != seed:
        raise ValueError("Checkpoint seed is incompatible with this run.")


def _atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    """Save through a sibling temporary file so failed writes preserve the target."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            torch.save(payload, temporary_file)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as input_file:
        return [
            {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
            for row in csv.DictReader(input_file)
        ]


def _json_config(config: ExperimentConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _run_manifest(config: ExperimentConfig, device: torch.device, train_count: int, validation_count: int) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "seed": config.seed,
        "git_commit": _git_commit(),
        "train_samples": train_count,
        "validation_samples": validation_count,
        "train_csv_name": config.train_csv.name,
        "validation_csv_name": config.validation_csv.name,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)


def _model_config(config: ExperimentConfig) -> dict[str, int]:
    return {"in_channels": 1, "out_channels": 1, "base_channels": config.base_channels, "depth": config.depth}


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _validate_config(config: ExperimentConfig) -> None:
    if config.epochs <= 0 or config.patience <= 0:
        raise ValueError("epochs and patience must be positive.")
    if config.image_size <= 0 or config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("image_size and batch_size must be positive; num_workers non-negative.")
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
