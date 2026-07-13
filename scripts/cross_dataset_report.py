"""Summarize fixed-threshold external prediction CSVs in one cross-dataset table."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia_ai.evaluation.core import calibration_metrics, discrimination_metrics, validate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Frozen validation-selected threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = []
    for path in args.inputs:
        predictions = validate_predictions(pd.read_csv(path))
        dataset = predictions["dataset"].iloc[0] if "dataset" in predictions else Path(path).stem
        rows.append({"dataset": dataset, "sample_count": len(predictions), **discrimination_metrics(predictions, args.threshold), **calibration_metrics(predictions)})
    pd.DataFrame(rows).to_csv(args.output, index=False)
