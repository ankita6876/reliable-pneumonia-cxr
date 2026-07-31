"""Unit tests for MC-dropout uncertainty helpers."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.analysis.uncertainty_utils import (
    binary_uncertainty,
    enable_mc_dropout,
    selective_prediction,
    uncertainty_summary,
)


def test_enable_mc_dropout_keeps_batch_norm_in_evaluation_mode() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(p=0.25))
    modules = enable_mc_dropout(model)
    assert modules == [("1", 0.25)]
    assert model[0].training is False
    assert model[1].training is True


def test_entropy_and_mutual_information_are_correct() -> None:
    values = binary_uncertainty(np.array([[0.2, 0.5], [0.2, 0.5]]))
    assert np.allclose(values["predictive_entropy"], values["expected_entropy"])
    assert np.allclose(values["mutual_information"], [0.0, 0.0])
    assert np.allclose(values["predictive_entropy"][1], np.log(2.0))


def test_uncertainty_summary_generation() -> None:
    frame = pd.DataFrame({"predictive_entropy": [0.1, 0.3], "mutual_information": [0.01, 0.03],
                          "mc_standard_deviation": [0.2, 0.4], "category": ["TP", "FP"]})
    summary = uncertainty_summary(frame, "category")
    assert set(summary["group"]) == {"TP", "FP"}
    assert summary.loc[summary.group == "TP", "predictive_entropy_mean"].item() == 0.1


def test_selective_prediction_calculations() -> None:
    frame = pd.DataFrame({"uncertainty": [0.3, 0.1, 0.2], "correctness": [0, 1, 1]})
    curve, aurc = selective_prediction(frame, "uncertainty")
    full = curve.loc[curve.requested_coverage_percent == 100].iloc[0]
    assert full.accuracy == 2 / 3
    assert curve.loc[curve.requested_coverage_percent == 10, "accuracy"].item() == 1.0
    assert 0.0 <= aurc <= 1.0
