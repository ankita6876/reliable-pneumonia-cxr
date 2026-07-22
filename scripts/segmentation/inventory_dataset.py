"""Inventory a lung-segmentation dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pneumonia_ai.segmentation.inventory import inventory_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory files in a lung-segmentation dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the segmentation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = inventory_dataset(args.dataset_root)
    payload = inventory.to_dict()

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
