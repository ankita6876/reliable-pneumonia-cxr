"""
Reproduce LCDR vs P7 PadChest publication artifacts.

Required inputs:
  1. corrected PadChest prediction CSV
  2. paired patient-cluster bootstrap CSV
  3. primary summary JSON

Frozen prediction SHA256:
7672fdd57f253c0aa08f79f7767705bfbab45ddcd919997065db11ba423c0d5e

LCDR checkpoint SHA256:
302728a6997d6ce5c1f30dd1098c05390fd29aad4f31f79e7cf68936652cf65f

P7 checkpoint SHA256:
bb3d82e4a13087efeb484ae9cd8478059c27aa006cbac5cefbe5c0c9a61157ac

Statistical protocol:
  - AUROC / AUPRC
  - paired patient-cluster bootstrap
  - 1000 replicates
  - seed 42
  - percentile 95% CI
  - AUROC non-inferiority margin = -0.01

PadChest labels are evaluation-only.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

EXPECTED_PREDICTION_SHA256 = (
    "7672fdd57f253c0aa08f79f7767705bfbab45ddcd919997065db11ba423c0d5e"
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_inputs(prediction_csv, bootstrap_csv, summary_json):
    prediction_csv = Path(prediction_csv)
    bootstrap_csv = Path(bootstrap_csv)
    summary_json = Path(summary_json)

    actual = sha256(prediction_csv)
    if actual != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError(
            f"Prediction SHA mismatch: {actual}"
        )

    predictions = pd.read_csv(prediction_csv)
    bootstrap = pd.read_csv(bootstrap_csv)

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return predictions, bootstrap, summary

# The frozen publication outputs in
# results/publication/lcdr_padchest and figures/lcdr_padchest
# document the exact results generated from these inputs.
#
# This helper intentionally does not contain or redistribute
# the large patient-level prediction artifact itself.
