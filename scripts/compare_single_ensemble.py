"""Create paired single-model versus deep-ensemble evaluation and figures."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia_ai.evaluation.core import compare_single_and_ensemble  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse paired comparison inputs without any model fitting or selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-predictions", required=True)
    parser.add_argument("--ensemble-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_single_and_ensemble(
        pd.read_csv(args.single_predictions),
        pd.read_csv(args.ensemble_predictions),
        args.output_dir,
        args.threshold,
        args.bootstrap_iterations,
        args.seed,
    )
