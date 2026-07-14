"""Build a compact metadata-only experiment registry."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia_ai.reporting.registry import build_registry  # noqa: E402


def main() -> None:
    """Parse registry arguments and write the discovered metadata table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registry = build_registry(args.inputs)
    registry.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
