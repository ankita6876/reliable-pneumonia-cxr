"""Train the first Montgomery U-Net experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from pneumonia_ai.segmentation.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    """Parse train/validation experiment options."""

    parser = argparse.ArgumentParser(description="Train a Montgomery lung U-Net.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run training using the requested configuration."""

    args = parse_args()
    run_experiment(ExperimentConfig(**vars(args)))


if __name__ == "__main__":
    main()
