"""Efficient development-only training with reproducible recovery state."""

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import time
from typing import Callable, Iterable

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.data.label_strategy import apply_label_strategy
from pneumonia_ai.training.seed import seed_worker


DEVELOPMENT_SPLITS = ("train", "validation")
UNCERTAIN_LABEL_STRATEGY = "ignore"


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and validation discrimination metrics for one epoch."""

    loss: float
    auroc: float | None = None
    auprc: float | None = None


def select_device() -> torch.device:
    """Choose CUDA when available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_is_enabled(requested: bool, device: torch.device) -> bool:
    """Enable AMP only for a CUDA device with CUDA actually available."""
    return requested and device.type == "cuda" and torch.cuda.is_available()


def calculate_pos_weight(labels: Iterable[float | int], enabled: bool = True) -> float | None:
    """Return negative-to-positive target mass after the selected strategy."""
    if not enabled:
        return None
    values = [float(label) for label in labels]
    positive_mass = sum(values)
    negative_mass = sum(1.0 - value for value in values)
    if positive_mass == 0.0 or negative_mass == 0.0:
        return 1.0
    return negative_mass / positive_mass


def count_raw_training_labels(manifest_path: Path | str) -> tuple[int, int]:
    """Return definite and uncertain counts from the original train split."""
    manifest = pd.read_csv(manifest_path)
    train_labels = pd.to_numeric(
        manifest.loc[manifest["split"] == "train", "pneumonia_label"], errors="coerce"
    )
    return int(train_labels.isin([0, 1]).sum()), int((train_labels == -1).sum())


def build_train_validation_datasets(
    root: Path | str,
    manifest_path: Path | str,
    output_dir: Path | str,
    train_transform: Callable[[object], object],
    validation_transform: Callable[[object], object],
    label_strategy: str = UNCERTAIN_LABEL_STRATEGY,
    uncertain_soft_target: float = 0.5,
    uncertain_sample_weight: float = 0.5,
    max_samples_per_split: int | None = None,
) -> tuple[CheXpertPneumoniaDataset, CheXpertPneumoniaDataset]:
    """Build strategy-specific training and definite-label validation datasets."""
    del output_dir  # Run directories retain only the prescribed final artifacts.
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest:
        raise ValueError("Split manifest is missing required column: split")
    train_records = manifest.loc[manifest["split"] == "train"]
    validation_records = manifest.loc[manifest["split"] == "validation"]
    training_records = apply_label_strategy(
        train_records,
        label_strategy,
        uncertain_soft_target=uncertain_soft_target,
        uncertain_sample_weight=uncertain_sample_weight,
    )
    definite_validation_records = validation_records.loc[
        pd.to_numeric(validation_records["pneumonia_label"], errors="coerce").isin([0, 1])
    ]
    validation_training_records = apply_label_strategy(
        definite_validation_records,
        label_strategy,
        uncertain_soft_target=uncertain_soft_target,
        uncertain_sample_weight=uncertain_sample_weight,
    )
    binary_records = pd.concat([training_records, validation_training_records])
    if max_samples_per_split is not None:
        binary_records = binary_records.groupby("split", group_keys=False).head(
            max_samples_per_split
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        filtered_manifest_path = Path(temporary_directory) / "development_manifest.csv"
        binary_records.to_csv(filtered_manifest_path, index=False)
        train_dataset = CheXpertPneumoniaDataset(
            root, filtered_manifest_path, "train", train_transform
        )
        validation_dataset = CheXpertPneumoniaDataset(
            root, filtered_manifest_path, "validation", validation_transform
        )
    return train_dataset, validation_dataset


def create_development_loaders(
    train_dataset: CheXpertPneumoniaDataset,
    validation_dataset: CheXpertPneumoniaDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create efficient reproducible loaders for train and validation only."""
    generator = torch.Generator().manual_seed(seed)
    common_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common_settings),
        DataLoader(validation_dataset, shuffle=False, **common_settings),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Run training batches with CUDA AMP when enabled."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = _batch_targets(batch, device)
        sample_weights = _batch_sample_weights(batch, labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(images).view(-1)
            loss = _weighted_mean_loss(criterion(logits, labels), sample_weights)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
    if batch_count == 0:
        raise ValueError("Training loader produced no batches.")
    return EpochMetrics(loss=total_loss / batch_count)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Run validation batches and calculate loss, AUROC, and AUPRC."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = _batch_targets(batch, device)
            raw_labels = _batch_raw_labels(batch, batch_labels, device)
            definite_mask = (raw_labels == 0.0) | (raw_labels == 1.0)
            if not definite_mask.any():
                continue
            batch_labels = batch_labels[definite_mask]
            sample_weights = _batch_sample_weights(batch, raw_labels, device)[definite_mask]
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                logits = model(images).view(-1)[definite_mask]
                loss = _weighted_mean_loss(criterion(logits, batch_labels), sample_weights)
            total_loss += loss.item()
            batch_count += 1
            labels.extend(batch_labels.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            if max_batches is not None and batch_count >= max_batches:
                break
    if batch_count == 0:
        raise ValueError("Validation loader produced no batches.")
    return EpochMetrics(
        loss=total_loss / batch_count,
        auroc=_safe_auroc(labels, probabilities),
        auprc=_safe_auprc(labels, probabilities),
    )


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: Path | str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    scheduler_factor: float,
    scheduler_patience: int,
    scheduler_min_lr: float,
    pos_weight: float | None,
    amp_requested: bool,
    configuration: dict[str, object],
    resume_checkpoint: Path | str | None = None,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Train on development data, retaining only the best atomic checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = device or select_device()
    amp_enabled = amp_is_enabled(amp_requested, device)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_path = output_path / "training_history.csv"
    history = _load_history(history_path)
    best_auroc = -math.inf
    start_epoch = 1
    checkpoint_path = output_path / "best_validation_auroc.pt"
    last_checkpoint_path = output_path / "last_checkpoint.pt"
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        _restore_checkpoint(
            resume_state, model, optimizer, scheduler, scaler, configuration
        )
        best_auroc = float(resume_state["best_validation_auroc"])
        start_epoch = int(resume_state["epoch"]) + 1
        history = [row for row in history if int(row["epoch"]) < start_epoch]

    no_improvement_epochs = 0
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        training_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, amp_enabled, max_batches
        )
        validation_metrics = validate_one_epoch(
            model, validation_loader, criterion, device, amp_enabled, max_batches
        )
        scheduler.step(validation_metrics.loss)
        duration_seconds = time.perf_counter() - epoch_start
        learning_rate_value = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate_value,
                "train_loss": training_metrics.loss,
                "validation_loss": validation_metrics.loss,
                "validation_auroc": validation_metrics.auroc,
                "validation_auprc": validation_metrics.auprc,
                "epoch_duration_seconds": duration_seconds,
            }
        )
        selection_auroc = validation_metrics.auroc
        comparison_auroc = -math.inf if selection_auroc is None else selection_auroc
        if comparison_auroc > best_auroc + early_stopping_min_delta:
            best_auroc = comparison_auroc
            no_improvement_epochs = 0
            _save_checkpoint_atomic(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_auroc=best_auroc,
                configuration=configuration,
            )
        else:
            no_improvement_epochs += 1
        _save_checkpoint_atomic(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_auroc=best_auroc,
            configuration=configuration,
        )
        if no_improvement_epochs >= early_stopping_patience:
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False)
    return history_frame


def _load_history(history_path: Path) -> list[dict[str, object]]:
    """Load prior compact history only when a run is being resumed."""
    return [] if not history_path.exists() else pd.read_csv(history_path).to_dict("records")


def _batch_targets(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    """Read a training target while accepting pre-framework test batches."""
    values = batch.get("target", batch["label"])
    return values.to(device, non_blocking=True).float().view(-1)


def _batch_raw_labels(
    batch: dict[str, object], targets: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Read raw labels, defaulting to targets for legacy definite-label batches."""
    values = batch.get("raw_label")
    return targets if values is None else values.to(device, non_blocking=True).float().view(-1)


def _batch_sample_weights(
    batch: dict[str, object], reference: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Read loss weights, defaulting to one for legacy definite-label batches."""
    values = batch.get("sample_weight")
    if values is None:
        return torch.ones_like(reference, device=device)
    return values.to(device, non_blocking=True).float().view(-1)


def _weighted_mean_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply per-sample weights to unreduced BCE loss before averaging."""
    return (losses * weights).sum() / weights.sum()


def _restore_checkpoint(
    checkpoint: dict[str, object],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    configuration: dict[str, object],
) -> None:
    """Restore recoverable state after verifying the effective configuration."""
    if checkpoint.get("configuration") != configuration:
        raise ValueError("Resume checkpoint configuration is incompatible with this run.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)


def _save_checkpoint_atomic(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_auroc: float,
    configuration: dict[str, object],
) -> None:
    """Atomically replace the sole best checkpoint after a meaningful improvement."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_auroc": best_validation_auroc,
        "configuration": configuration,
    }
    temporary_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.tmp{checkpoint_path.suffix}"
    )
    try:
        if temporary_path.exists():
            temporary_path.unlink()
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _safe_auroc(labels: list[float], probabilities: list[float]) -> float | None:
    """Return AUROC when both binary classes are represented."""
    return None if len(set(labels)) < 2 else float(roc_auc_score(labels, probabilities))


def _safe_auprc(labels: list[float], probabilities: list[float]) -> float | None:
    """Return AUPRC when both binary classes are represented."""
    return None if len(set(labels)) < 2 else float(average_precision_score(labels, probabilities))
