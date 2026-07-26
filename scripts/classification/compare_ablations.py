"""Create a compact metric comparison across classifier input-mode ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode  # noqa: E402


METRICS = ("auc", "pr_auc", "f1", "ece", "brier", "accuracy", "recall", "specificity")


def parse_args() -> argparse.Namespace:
    """Parse the ablation root and comparison output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path)
    return parser.parse_args()


def build_comparison(output_directory: Path, comparison_csv: Path | None = None) -> pd.DataFrame:
    """Read all mode metrics and persist a consistently ordered comparison CSV."""
    rows: list[dict[str, object]] = []
    for mode in InputMode:
        metrics_path = output_directory / mode.value / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing ablation metrics: {metrics_path}")
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid metrics JSON: {metrics_path}")
        missing = [name for name in METRICS if name not in metrics]
        if missing:
            raise ValueError(f"Metrics file {metrics_path} is missing: {', '.join(missing)}")
        rows.append({"mode": mode.value, **{name: metrics[name] for name in METRICS}})
    comparison = pd.DataFrame(rows, columns=("mode", *METRICS))
    destination = comparison_csv or output_directory / "comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination, index=False)
    return comparison


def main() -> None:
    args = parse_args()
    comparison = build_comparison(args.output_directory, args.comparison_csv)
    print(f"Wrote {len(comparison)} modes to {args.comparison_csv or args.output_directory / 'comparison.csv'}")


if __name__ == "__main__":
    main()
