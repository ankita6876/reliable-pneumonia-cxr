"""Evaluate a trained U-Net checkpoint on one explicit split."""

from __future__ import annotations

import argparse
from pathlib import Path

from pneumonia_ai.segmentation.evaluation import evaluate_checkpoint


DEFAULT_SPLIT = Path(
    r"C:\Research\datasets\lung_segmentation\montgomery\processed\splits\test.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""

    parser = argparse.ArgumentParser(description="Evaluate a Montgomery U-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--qualitative-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run requested split evaluation and print aggregate results."""

    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split_csv=args.split_csv,
        output_directory=args.output_directory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        threshold=args.threshold,
        qualitative_count=args.qualitative_count,
    )
    print(metrics)


if __name__ == "__main__":
    main()
