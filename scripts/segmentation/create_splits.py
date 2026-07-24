"""Create deterministic Montgomery segmentation train/validation/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from pneumonia_ai.segmentation.splitting import DEFAULT_SEED, create_segmentation_splits


DEFAULT_METADATA_PATH = Path(
    r"C:\Research\datasets\lung_segmentation\montgomery\processed\metadata.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse split-generation command line options."""

    parser = argparse.ArgumentParser(
        description="Create deterministic Montgomery segmentation data splits."
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_METADATA_PATH.parent / "splits",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Write train, validation, and test manifests and print their locations."""

    args = parse_args()
    splits = create_segmentation_splits(
        args.metadata_path, args.output_directory, seed=args.seed
    )
    for name, split in splits.items():
        print(f"Wrote {name} split ({len(split)} samples): {args.output_directory / f'{name}.csv'}")


if __name__ == "__main__":
    main()
