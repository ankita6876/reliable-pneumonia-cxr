"""Train one original, hard-masked, or lung-crop classifier ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.ablation import AblationConfig, run_ablation  # noqa: E402
from pneumonia_ai.classification.segmentation_guided import InputMode  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse a complete ablation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=[mode.value for mode in InputMode], required=True)
    parser.add_argument(
        "--splits-csv",
        type=Path,
        required=True,
        help="Patient-level manifest with train, validation, and test split rows.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory against which relative image_path values are resolved.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--classifier-image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the requested ablation and report its output location."""
    args = parse_args()
    directory = run_ablation(AblationConfig(**vars(args)))
    print(f"Completed {args.input_mode} ablation: {directory}")


if __name__ == "__main__":
    main()
