"""Checkpoint evaluation, prediction export, and qualitative visualization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

import cv2
import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as pyplot

from pneumonia_ai.segmentation.dataset import MontgomerySegmentationDataset
from pneumonia_ai.segmentation.experiment import _resolve_device, _validate_checkpoint
from pneumonia_ai.segmentation.losses import BCEWithLogitsDiceLoss
from pneumonia_ai.segmentation.metrics import per_sample_metrics
from pneumonia_ai.segmentation.model import UNet
from pneumonia_ai.segmentation.training import build_dataloader
from pneumonia_ai.segmentation.transforms import EvaluationTransform


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path | str,
    split_csv: Path | str,
    output_directory: Path | str,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "cpu",
    threshold: float = 0.5,
    qualitative_count: int = 8,
) -> dict[str, float | int]:
    """Evaluate one requested split and save reproducible prediction artifacts."""

    if batch_size <= 0 or num_workers < 0 or qualitative_count < 0:
        raise ValueError("batch_size must be positive; worker and figure counts non-negative.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_file, map_location="cpu", weights_only=False
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model configuration is missing or invalid.")
    _validate_checkpoint(checkpoint, model_config, checkpoint.get("seed"))
    model = UNet(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    resolved_device = _resolve_device(device)
    model.to(resolved_device).eval()
    training_config = checkpoint.get("training_config", {})
    image_size = int(training_config.get("image_size", 128))
    loss_function = BCEWithLogitsDiceLoss(
        float(training_config.get("bce_weight", 0.5)),
        float(training_config.get("dice_weight", 0.5)),
    )
    dataset = MontgomerySegmentationDataset(
        split_csv, image_size=image_size, transform=EvaluationTransform()
    )
    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    output_path = Path(output_directory).expanduser()
    mask_directory = output_path / "prediction_masks"
    mask_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    total_loss = 0.0
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    sample_count = 0
    for images, masks, metadata in loader:
        images = images.to(resolved_device)
        masks = masks.to(resolved_device)
        logits = model(images)
        loss = loss_function(logits, masks)
        values = per_sample_metrics(logits, masks, threshold=threshold)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).to(dtype=torch.uint8)
        batch_size_actual = images.shape[0]
        total_loss += loss.item() * batch_size_actual
        for index in range(batch_size_actual):
            image_id = str(metadata["image_id"][index])
            row: dict[str, float | str] = {"image_id": image_id}
            for name, values_per_sample in values.items():
                value = values_per_sample[index].item()
                row[name] = value
                totals[name] += value
            rows.append(row)
            _save_binary_mask(mask_directory / f"{_portable_name(image_id)}.png", predictions[index, 0])
        sample_count += batch_size_actual
    if sample_count == 0:
        raise ValueError("Evaluation split must contain at least one sample.")
    aggregate = {
        "loss": total_loss / sample_count,
        "dice": totals["dice"] / sample_count,
        "iou": totals["iou"] / sample_count,
        "precision": totals["precision"] / sample_count,
        "recall": totals["sensitivity"] / sample_count,
        "sensitivity": totals["sensitivity"] / sample_count,
        "specificity": totals["specificity"] / sample_count,
        "samples": sample_count,
    }
    _write_per_sample_metrics(output_path / "per_sample_metrics.csv", rows)
    with (output_path / "aggregate_metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(aggregate, output_file, indent=2, sort_keys=True)
    _save_qualitative_figures(
        model,
        dataset,
        output_path / "qualitative",
        resolved_device,
        threshold,
        qualitative_count,
    )
    return aggregate


def _save_binary_mask(path: Path, prediction: torch.Tensor) -> None:
    """Persist a `{0, 255}` single-channel PNG prediction."""

    array = prediction.detach().to(device="cpu", dtype=torch.uint8).numpy() * 255
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Unable to write predicted mask: {path}")


@torch.inference_mode()
def _save_qualitative_figures(
    model: UNet,
    dataset: MontgomerySegmentationDataset,
    output_directory: Path,
    device: torch.device,
    threshold: float,
    qualitative_count: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for index in range(min(qualitative_count, len(dataset))):
        image, mask, metadata = dataset[index]
        probabilities = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu()
        predicted_mask = probabilities >= threshold
        image_array = image[0].numpy()
        figure, axes = pyplot.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(image_array, cmap="gray")
        axes[0].set_title("X-ray")
        axes[1].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")
        axes[2].imshow(probabilities.numpy(), cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Probability")
        axes[3].imshow(predicted_mask.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Prediction")
        axes[4].imshow(image_array, cmap="gray")
        axes[4].imshow(predicted_mask.numpy(), cmap="spring", alpha=0.4, vmin=0, vmax=1)
        axes[4].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(
            output_directory / f"{_portable_name(metadata['image_id'])}.png", dpi=150
        )
        pyplot.close(figure)


def _write_per_sample_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_id", "dice", "iou", "precision", "sensitivity", "specificity"])
        writer.writeheader()
        writer.writerows(rows)


def _portable_name(value: str) -> str:
    """Return a portable deterministic filename stem."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sample"
