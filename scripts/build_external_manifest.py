"""Create a standard root-relative external-validation manifest from source metadata."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia_ai.data.multidataset import build_standard_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chexpert", "nih", "mimic"), required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="external")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_standard_manifest(args.dataset, args.metadata, args.root, args.split).to_csv(args.output, index=False)
