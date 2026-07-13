"""Fit a leakage-safe temperature scaler from validation prediction CSV data."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.evaluation.core import fit_temperature  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation-predictions", required=True)
parser.add_argument("--output", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    result = fit_temperature(pd.read_csv(args.validation_predictions))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
