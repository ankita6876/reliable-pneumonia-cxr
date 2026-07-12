"""Train the initial DenseNet121 CheXpert baseline on train and validation only."""

import argparse
from pathlib import Path
import sys

from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.training.engine import (  # noqa: E402
    build_train_validation_datasets,
    create_development_loaders,
    run_training,
)
from pneumonia_ai.training.seed import seed_everything  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    return parser.parse_args()


def _load_config(config_path: Path | str) -> dict[str, object]:
    """Load YAML configuration from the requested path."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return allowed baseline training and deterministic validation transforms."""
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


def _create_model_from_config(
    model_config: dict[str, object], dry_run: bool
):
    """Construct the YAML-selected backbone without downloads during dry runs."""
    model_name = model_config.get("name")
    if not isinstance(model_name, str):
        raise ValueError("Configuration model.name must be a string.")
    pretrained = model_config.get("pretrained")
    if not isinstance(pretrained, bool):
        raise ValueError("Configuration model.pretrained must be a boolean.")
    return create_model(model_name, pretrained=False if dry_run else pretrained)


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
    seed_everything(seed)
    train_transform, validation_transform = _transforms(image_size)
    try:
        train_dataset, validation_dataset = build_train_validation_datasets(
            args.root,
            args.manifest,
            output_dir,
            train_transform,
            validation_transform,
            max_samples_per_split=2 if args.dry_run else None,
        )
        train_loader, validation_loader = create_development_loaders(
            train_dataset,
            validation_dataset,
            batch_size=min(int(training_config["batch_size"]), 2) if args.dry_run else int(training_config["batch_size"]),
            num_workers=int(training_config["num_workers"]),
            seed=seed,
        )
        model = _create_model_from_config(model_config, args.dry_run)
        history = run_training(
            model,
            train_loader,
            validation_loader,
            output_dir,
            epochs=1 if args.dry_run else int(training_config["epochs"]),
            learning_rate=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            max_batches=1 if args.dry_run else None,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Training error: {error}", file=sys.stderr)
        return 1

    print(history.to_string(index=False))
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
