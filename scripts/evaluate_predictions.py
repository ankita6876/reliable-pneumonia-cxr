"""Evaluate validation predictions, then apply frozen choices to optional test predictions."""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.evaluation.core import evaluate_predictions  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation-predictions", required=True)
parser.add_argument("--test-predictions")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--threshold-method", choices=("youden", "max_f1", "fixed_0.5"), default="youden")
parser.add_argument("--bootstrap-iterations", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)

if __name__ == "__main__":
    args = parser.parse_args()
    evaluate_predictions(pd.read_csv(args.validation_predictions), pd.read_csv(args.test_predictions) if args.test_predictions else None, args.output_dir, args.threshold_method, args.bootstrap_iterations, args.seed)
