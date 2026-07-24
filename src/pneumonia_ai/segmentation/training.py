"""CPU-safe training primitives for binary lung segmentation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader, Dataset

from pneumonia_ai.segmentation.metrics import segmentation_metrics


@dataclass(frozen=True)
class EpochSummary:
    """Aggregated loss and segmentation metrics from one epoch."""

    loss: float
    dice: float
    iou: float
    sensitivity: float
    specificity: float
    precision: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return the summary in a logging-friendly format."""

        return asdict(self)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_dataloader(
    dataset: Dataset[object],
    batch_size: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader[object]:
    """Construct a deterministic DataLoader with CPU-safe worker defaults."""

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )


def build_optimizer(
    model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 0.0
) -> Optimizer:
    """Build the default AdamW optimizer."""

    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer, step_size: int = 10, gamma: float = 0.5
) -> LRScheduler:
    """Build a StepLR scheduler for epoch-level stepping."""

    if step_size <= 0 or not 0.0 < gamma <= 1.0:
        raise ValueError("step_size must be positive and gamma must be in (0, 1].")
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[object],
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device | str = "cpu",
    gradient_clip_norm: float | None = None,
    use_amp: bool = False,
) -> EpochSummary:
    """Train on one loader epoch without consulting validation or test data."""

    model.train()
    return _run_epoch(
        model,
        loader,
        loss_function,
        device=torch.device(device),
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        use_amp=use_amp,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[object],
    loss_function: nn.Module,
    device: torch.device | str = "cpu",
) -> EpochSummary:
    """Evaluate one validation loader epoch without updating parameters."""

    model.eval()
    return _run_epoch(
        model,
        loader,
        loss_function,
        device=torch.device(device),
        optimizer=None,
        gradient_clip_norm=None,
        use_amp=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: Iterable[object],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    gradient_clip_norm: float | None,
    use_amp: bool,
) -> EpochSummary:
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    model.to(device)
    amp_enabled = use_amp and device.type == "cuda" and torch.cuda.is_available()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0, "precision": 0.0}
    sample_count = 0
    for batch in loader:
        images, masks = _extract_images_and_masks(batch)
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
        with autocast_context:
            logits = model(images)
            loss = loss_function(logits, masks)
        if optimizer is not None:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        metrics = segmentation_metrics(logits.detach(), masks)
        batch_size = images.shape[0]
        totals["loss"] += loss.detach().item() * batch_size
        for name, value in metrics.to_dict().items():
            totals[name] += value * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Epoch loader must contain at least one sample.")
    return EpochSummary(
        **{name: total / sample_count for name, total in totals.items()},
        samples=sample_count,
    )


def _extract_images_and_masks(batch: object) -> tuple[Tensor, Tensor]:
    """Extract tensors from the dataset's ``(image, mask, metadata)`` batches."""

    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Each loader batch must contain image and mask tensors.")
    images, masks = batch[0], batch[1]
    if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
        raise TypeError("Batch image and mask entries must be PyTorch tensors.")
    return images, masks
