"""Train and validate the initial DenseNet121 baseline on development splits only."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.data.label_strategy import apply_label_strategy
from pneumonia_ai.training.seed import seed_worker


DEVELOPMENT_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and validation discrimination metrics for one epoch."""

    loss: float
    auroc: float | None = None
    auprc: float | None = None


def select_device() -> torch.device:
    """Choose CUDA when available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_validation_datasets(
    root: Path | str,
    manifest_path: Path | str,
    output_dir: Path | str,
    train_transform: Callable[[object], object],
    validation_transform: Callable[[object], object],
    max_samples_per_split: int | None = None,
) -> tuple[CheXpertPneumoniaDataset, CheXpertPneumoniaDataset]:
    """Apply ``ignore`` to development rows before creating train/validation datasets."""
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest:
        raise ValueError("Split manifest is missing required column: split")
    development_records = manifest.loc[manifest["split"].isin(DEVELOPMENT_SPLITS)]
    binary_records = apply_label_strategy(development_records, "ignore")
    if max_samples_per_split is not None:
        binary_records = binary_records.groupby("split", group_keys=False).head(
            max_samples_per_split
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filtered_manifest_path = output_path / "manifest_ignore_development.csv"
    binary_records.to_csv(filtered_manifest_path, index=False)
    return (
        CheXpertPneumoniaDataset(root, filtered_manifest_path, "train", train_transform),
        CheXpertPneumoniaDataset(
            root, filtered_manifest_path, "validation", validation_transform
        ),
    )


def create_development_loaders(
    train_dataset: CheXpertPneumoniaDataset,
    validation_dataset: CheXpertPneumoniaDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible train and validation DataLoaders, never a test loader."""
    generator = torch.Generator().manual_seed(seed)
    common_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
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
    max_batches: int | None = None,
) -> EpochMetrics:
    """Run training batches and return their mean BCE-with-logits loss."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
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
            images = batch["image"].to(device)
            batch_labels = batch["label"].to(device).float().view(-1)
            logits = model(images).view(-1)
            loss = criterion(logits, batch_labels)
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
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Train only on train, select only on validation, and persist reproducible outputs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = device or select_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float | int | None]] = []
    best_auroc = -math.inf
    checkpoint_path = output_path / "best_validation_auroc.pt"
    for epoch in range(1, epochs + 1):
        training_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches
        )
        validation_metrics = validate_one_epoch(
            model, validation_loader, criterion, device, max_batches
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_metrics.loss,
                "validation_loss": validation_metrics.loss,
                "validation_auroc": validation_metrics.auroc,
                "validation_auprc": validation_metrics.auprc,
            }
        )
        selection_auroc = validation_metrics.auroc
        comparison_auroc = -math.inf if selection_auroc is None else selection_auroc
        if comparison_auroc >= best_auroc:
            best_auroc = comparison_auroc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "auroc": selection_auroc},
                checkpoint_path,
            )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_path / "training_history.csv", index=False)
    return history_frame


def _safe_auroc(labels: list[float], probabilities: list[float]) -> float | None:
    """Return AUROC when both binary classes are represented."""
    return None if len(set(labels)) < 2 else float(roc_auc_score(labels, probabilities))


def _safe_auprc(labels: list[float], probabilities: list[float]) -> float | None:
    """Return AUPRC when both binary classes are represented."""
    return None if len(set(labels)) < 2 else float(average_precision_score(labels, probabilities))
