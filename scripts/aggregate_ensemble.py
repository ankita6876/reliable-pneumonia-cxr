"""Aggregate aligned deterministic-model prediction CSVs into a deep ensemble."""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.evaluation.core import aggregate_ensemble  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    aggregate_ensemble([pd.read_csv(path) for path in args.inputs]).to_csv(args.output, index=False)
