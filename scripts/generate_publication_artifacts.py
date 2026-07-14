"""Generate compact publication tables and figures from a registry CSV."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia_ai.reporting.registry import generate_publication_artifacts  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--registry", required=True)
parser.add_argument("--output-dir", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    generate_publication_artifacts(pd.read_csv(args.registry), args.output_dir)
