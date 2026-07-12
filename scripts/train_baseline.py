"""Train the initial DenseNet121 CheXpert baseline on train and validation only."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import torch
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.training.engine import (  # noqa: E402
    UNCERTAIN_LABEL_STRATEGY,
    amp_is_enabled,
    build_train_validation_datasets,
    calculate_pos_weight,
    count_raw_training_labels,
    create_development_loaders,
    run_training,
    select_device,
)
from pneumonia_ai.training.experiment import (  # noqa: E402
    create_run_directory,
    write_resolved_config,
    write_run_manifest,
)
from pneumonia_ai.training.seed import seed_everything  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PREPROCESSING = "imagenet"
XRV_PREPROCESSING = "torchxrayvision"


def parse_args() -> argparse.Namespace:
    """Parse baseline-training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="CheXpert dataset root.")
    parser.add_argument("--manifest", required=True, help="Patient-level split manifest CSV.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline_densenet121.yaml"),
        help="YAML baseline configuration path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one batch per development split.")
    parser.add_argument(
        "--resume",
        help="Existing run directory or best checkpoint to resume after configuration validation.",
    )
    return parser.parse_args()


def _load_config(config_path: Path | str) -> dict[str, object]:
    """Load YAML configuration from the requested path."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _xrv_normalize(image: torch.Tensor) -> torch.Tensor:
    """Map an 8-bit tensor to TorchXRayVision's [-1024, 1024] convention."""
    return image.mul(2048.0).sub(1024.0)


def _transforms(
    image_size: int, preprocessing: str = IMAGENET_PREPROCESSING
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return model-configured training and deterministic validation transforms."""
    if preprocessing == IMAGENET_PREPROCESSING:
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ToTensor(),
                normalize,
            ]
        )
        validation_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, validation_transform
    if preprocessing != XRV_PREPROCESSING:
        raise ValueError(
            "Configuration model.preprocessing must be 'imagenet' or 'torchxrayvision'."
        )
    if image_size != 224:
        raise ValueError("TorchXRayVision DenseNet121 requires image_size 224.")
    xrv_base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    train_transform = transforms.Compose(
        [
            *xrv_base,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Lambda(_xrv_normalize),
        ]
    )
    validation_transform = transforms.Compose(
        [*xrv_base, transforms.ToTensor(), transforms.Lambda(_xrv_normalize)]
    )
    return train_transform, validation_transform


def _create_model_from_config(
    model_config: dict[str, object], dry_run: bool, resume: bool = False
):
    """Construct the YAML-selected backbone without downloads during dry runs."""
    model_name = model_config.get("name")
    if not isinstance(model_name, str):
        raise ValueError("Configuration model.name must be a string.")
    pretrained = model_config.get("pretrained")
    if not isinstance(pretrained, bool):
        raise ValueError("Configuration model.pretrained must be a boolean.")
    return create_model(model_name, pretrained=False if dry_run or resume else pretrained)


def _checkpoint_configuration(config: dict[str, object], dry_run: bool) -> dict[str, object]:
    """Return the reproducibility-relevant configuration stored in checkpoints."""
    checkpoint_config = deepcopy(config)
    checkpoint_config["dry_run"] = dry_run
    if dry_run:
        checkpoint_config["model"]["pretrained"] = False
        checkpoint_config["training"]["epochs"] = 1
        checkpoint_config["training"]["batch_size"] = min(
            int(checkpoint_config["training"]["batch_size"]), 2
        )
    return checkpoint_config


def _label_settings(config: dict[str, object]) -> tuple[str, float, float]:
    """Read the predefined uncertainty strategy and its soft-label settings."""
    label_config = config.get("label_strategy", {"name": UNCERTAIN_LABEL_STRATEGY})
    if not isinstance(label_config, dict):
        raise ValueError("Configuration label_strategy must be a mapping.")
    strategy = label_config.get("name", UNCERTAIN_LABEL_STRATEGY)
    soft_target = label_config.get("uncertain_soft_target", 0.5)
    sample_weight = label_config.get("uncertain_sample_weight", 0.5)
    if not isinstance(strategy, str):
        raise ValueError("Configuration label_strategy.name must be a string.")
    return strategy, float(soft_target), float(sample_weight)


def _resume_checkpoint(path: str) -> Path:
    """Resolve a user-supplied run directory or its sole best checkpoint."""
    candidate = Path(path)
    checkpoint = candidate / "best_validation_auroc.pt" if candidate.is_dir() else candidate
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def _directory_size_bytes(directory: Path) -> int:
    """Return the compact persistent artifact size for user-facing reporting."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def main() -> int:
    """Run the configurable baseline or its bounded dry-run verification."""
    args = parse_args()
    config = _load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise ValueError("Configuration must contain model and training mappings.")
    seed = int(config["seed"])
    output_dir = Path(str(config["output_dir"]))
    image_size = int(training_config["image_size"])
    label_strategy, uncertain_soft_target, uncertain_sample_weight = _label_settings(config)
    model_name = model_config.get("name")
    pretrained = model_config.get("pretrained")
    if not isinstance(model_name, str) or not isinstance(pretrained, bool):
        raise ValueError("Configuration model.name and model.pretrained are required.")
    effective_batch_size = (
        min(int(training_config["batch_size"]), 2)
        if args.dry_run
        else int(training_config["batch_size"])
    )
    effective_epochs = 1 if args.dry_run else int(training_config["epochs"])
    effective_pretrained = False if args.dry_run or args.resume else pretrained
    resume_checkpoint = _resume_checkpoint(args.resume) if args.resume else None
    if resume_checkpoint is None:
        run_directory, run_id, timestamp = create_run_directory(output_dir, model_name)
        write_resolved_config(run_directory, config, args.dry_run)
    else:
        run_directory = resume_checkpoint.parent
        saved_config_path = run_directory / "config.yaml"
        if not saved_config_path.is_file():
            raise ValueError("Resume run directory is missing config.yaml.")
        run_id = "resumed"
        timestamp = "resumed"
    preprocessing = model_config.get("preprocessing", IMAGENET_PREPROCESSING)
    if not isinstance(preprocessing, str):
        raise ValueError("Configuration model.preprocessing must be a string.")
    seed_everything(seed)
    train_transform, validation_transform = _transforms(image_size, preprocessing)
    try:
        train_dataset, validation_dataset = build_train_validation_datasets(
            args.root,
            args.manifest,
            run_directory,
            train_transform,
            validation_transform,
            label_strategy=label_strategy,
            uncertain_soft_target=uncertain_soft_target,
            uncertain_sample_weight=uncertain_sample_weight,
            max_samples_per_split=2 if args.dry_run else None,
        )
        train_loader, validation_loader = create_development_loaders(
            train_dataset,
            validation_dataset,
            batch_size=effective_batch_size,
            num_workers=int(training_config["num_workers"]),
            seed=seed,
        )
        model = _create_model_from_config(model_config, args.dry_run, resume_checkpoint is not None)
        device = select_device()
        amp_enabled = amp_is_enabled(bool(training_config["amp_enabled"]), device)
        pos_weight = calculate_pos_weight(
            train_dataset._targets, bool(training_config["class_weighting"])
        )
        definite_training_count, uncertain_training_count = count_raw_training_labels(
            args.manifest
        )
        checkpoint_configuration = _checkpoint_configuration(config, args.dry_run)
        if resume_checkpoint is None:
            write_run_manifest(
                run_directory,
                run_id=run_id,
                timestamp=timestamp,
                model_name=model_name,
                pretrained=effective_pretrained,
                uncertain_label_strategy=label_strategy,
                seed=seed,
                image_size=image_size,
                batch_size=effective_batch_size,
                learning_rate=float(training_config["learning_rate"]),
                epoch_count=effective_epochs,
                device=device,
                split_manifest_path=args.manifest,
                train_sample_count=len(train_dataset),
                validation_sample_count=len(validation_dataset),
                dry_run=args.dry_run,
                amp_enabled=amp_enabled,
                pos_weight=pos_weight,
                uncertain_soft_target=uncertain_soft_target,
                uncertain_sample_weight=uncertain_sample_weight,
                definite_training_sample_count=definite_training_count,
                uncertain_training_sample_count=uncertain_training_count,
                effective_training_sample_count=len(train_dataset),
                definite_validation_sample_count=len(validation_dataset),
            )
        history = run_training(
            model,
            train_loader,
            validation_loader,
            run_directory,
            epochs=effective_epochs,
            learning_rate=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            early_stopping_patience=int(training_config["early_stopping_patience"]),
            early_stopping_min_delta=float(training_config["early_stopping_min_delta"]),
            scheduler_factor=float(training_config["scheduler_factor"]),
            scheduler_patience=int(training_config["scheduler_patience"]),
            scheduler_min_lr=float(training_config["scheduler_min_lr"]),
            pos_weight=pos_weight,
            amp_requested=bool(training_config["amp_enabled"]),
            configuration=checkpoint_configuration,
            resume_checkpoint=resume_checkpoint,
            device=device,
            max_batches=1 if args.dry_run else None,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Training error: {error}", file=sys.stderr)
        return 1

    print(history.to_string(index=False))
    print(f"Outputs: {run_directory}")
    print(f"Final run-directory size: {_directory_size_bytes(run_directory)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
