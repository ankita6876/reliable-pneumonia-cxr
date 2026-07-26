"""Reproducible classification ablations for segmentation-guided inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd
import torch
from torchvision import transforms

from pneumonia_ai.classification.segmentation_guided import InputMode
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.models.factory import create_model
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.training.engine import (
    UNCERTAIN_LABEL_STRATEGY,
    calculate_pos_weight,
    create_development_loaders,
    run_training,
)
from pneumonia_ai.training.seed import seed_everything


@dataclass(frozen=True)
class AblationConfig:
    """Fully specified classifier ablation settings."""

    input_mode: str
    train_csv: Path
    validation_csv: Path
    output_directory: Path
    segmentation_checkpoint: Path | None = None
    checkpoint: Path | None = None
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    classifier_image_size: int = 224
    num_workers: int = 0
    lung_crop_padding: int = 0
    mask_threshold: float = 0.5


def run_ablation(config: AblationConfig) -> Path:
    """Run one mode and return its mode-specific output directory.

    CSV image paths are interpreted relative to the training CSV's directory,
    matching portable CheXpert manifests. Both CSVs must use that same root.
    """
    mode = _validate_config(config)
    seed_everything(config.seed)
    run_directory = config.output_directory / mode.value
    run_directory.mkdir(parents=True, exist_ok=True)
    segmenter = (
        FrozenLungSegmenter(config.segmentation_checkpoint, config.device)
        if mode is not InputMode.ORIGINAL
        else None
    )
    cache = MaskCache(run_directory / "mask_cache") if segmenter is not None else None
    train_transform, validation_transform = _transforms(config.classifier_image_size)
    effective_config: dict[str, object] = {
        "input_mode": mode.value,
        "model": "densenet121",
        "classifier_image_size": config.classifier_image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "mask_threshold": config.mask_threshold,
        "lung_crop_padding": config.lung_crop_padding,
        "segmentation_checkpoint": _checkpoint_identity(config.segmentation_checkpoint),
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = _combined_manifest(
            config.train_csv, config.validation_csv, Path(temporary_directory)
        )
        root = config.train_csv.parent.resolve()
        train_dataset = CheXpertPneumoniaDataset(
            root, manifest_path, "train", train_transform, mode, segmenter, cache,
            config.mask_threshold, config.lung_crop_padding, config.classifier_image_size,
        )
        validation_dataset = CheXpertPneumoniaDataset(
            root, manifest_path, "validation", validation_transform, mode, segmenter, cache,
            config.mask_threshold, config.lung_crop_padding, config.classifier_image_size,
        )
        train_loader, validation_loader = create_development_loaders(
            train_dataset, validation_dataset, config.batch_size, config.num_workers, config.seed
        )
        model = create_model("densenet121", pretrained=False)
        run_training(
            model, train_loader, validation_loader, run_directory, config.epochs,
            config.learning_rate, config.weight_decay, 10, 0.0, 0.5, 2, 1e-7,
            calculate_pos_weight(train_dataset._targets.tolist()), False, effective_config,
            config.checkpoint, torch.device(config.device),
        )
    shutil.copy2(run_directory / "training_history.csv", run_directory / "history.csv")
    shutil.copy2(run_directory / "best_validation_auroc.pt", run_directory / "best_checkpoint.pt")
    _write_metadata(run_directory, effective_config, config)
    return run_directory


def _validate_config(config: AblationConfig) -> InputMode:
    try:
        mode = InputMode(config.input_mode)
    except ValueError as error:
        raise ValueError("input_mode must be original, hard_masked, or lung_crop.") from error
    if mode is not InputMode.ORIGINAL and config.segmentation_checkpoint is None:
        raise ValueError("hard_masked and lung_crop modes require segmentation_checkpoint.")
    if config.batch_size <= 0 or config.epochs <= 0 or config.classifier_image_size <= 0:
        raise ValueError("batch_size, epochs, and classifier_image_size must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.num_workers < 0:
        raise ValueError("learning_rate must be positive; weight_decay and num_workers non-negative.")
    return mode


def _combined_manifest(train_csv: Path, validation_csv: Path, directory: Path) -> Path:
    train, validation = pd.read_csv(train_csv), pd.read_csv(validation_csv)
    for frame, split in ((train, "train"), (validation, "validation")):
        if "image_path" not in frame:
            raise ValueError("Ablation CSVs must contain image_path.")
        frame["split"] = split
    path = directory / "ablation_manifest.csv"
    pd.concat([train, validation], ignore_index=True).to_csv(path, index=False)
    return path


def _transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    common = [
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(_to_rgb),
        transforms.ToTensor(),
        normalize,
    ]
    return (
        transforms.Compose([transforms.RandomHorizontalFlip(), *common]),
        transforms.Compose(common),
    )


def _checkpoint_identity(path: Path | None) -> str | None:
    return None if path is None else path.name


def _to_rgb(image: object) -> object:
    """Make guided grayscale images compatible with RGB classifier backbones."""
    return image.convert("RGB") if hasattr(image, "convert") else image


def _write_metadata(directory: Path, effective: dict[str, object], config: AblationConfig) -> None:
    (directory / "resolved_config.json").write_text(json.dumps(effective, indent=2, sort_keys=True))
    manifest: dict[str, Any] = {
        "input_mode": effective["input_mode"],
        "seed": config.seed,
        "device": config.device,
        "label_strategy": UNCERTAIN_LABEL_STRATEGY,
        "segmentation_checkpoint": effective["segmentation_checkpoint"],
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
