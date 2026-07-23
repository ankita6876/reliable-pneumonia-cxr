"""Build merged masks and metadata for the Montgomery dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pneumonia_ai.segmentation.dataset_builder import build_montgomery_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the Montgomery lung-segmentation dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing CXR_png and ManualMask.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the dataset and report the written metadata location."""

    args = parse_args()
    metadata = build_montgomery_dataset(args.dataset_root)
    metadata_path = (
        args.dataset_root.expanduser().resolve() / "processed" / "metadata.csv"
    )
    print(f"Built {len(metadata)} Montgomery samples: {metadata_path}")


if __name__ == "__main__":
    main()
