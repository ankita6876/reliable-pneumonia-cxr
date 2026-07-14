"""Synthetic, CPU-only tests for leakage-safe evaluation utilities."""
# ruff: noqa: E501
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import pneumonia_ai.evaluation.core as evaluation_core  # noqa: E402
from pneumonia_ai.evaluation.core import (  # noqa: E402
    add_deterministic_uncertainty,
    aggregate_ensemble,
    apply_temperature,
    bootstrap_confidence_intervals,
    calibration_metrics,
    compare_single_and_ensemble,
    delong_auroc_test,
    discrimination_metrics,
    evaluate_predictions,
    failure_detection,
    fit_temperature,
    selective_prediction,
    select_threshold,
    validate_predictions,
)


def _predictions(split: str = "validation") -> pd.DataFrame:
    probabilities = np.array([.05, .15, .25, .4, .6, .75, .85, .95])
    return pd.DataFrame({"patient_id":[f"p{i//2}" for i in range(8)], "study_id":[f"s{i}" for i in range(8)], "image_path":[f"images/{i}.png" for i in range(8)], "original_label":[0,0,0,0,1,1,1,1], "binary_target":[0,0,0,0,1,1,1,1], "logit":np.log(probabilities/(1-probabilities)), "probability":probabilities, "predicted_class":(probabilities>=.5).astype(int), "split":split, "model_name":"toy", "run_id":"run"})


def test_schema_metrics_threshold_and_calibration() -> None:
    frame = _predictions()
    validate_predictions(frame)
    metrics = discrimination_metrics(frame, .5)
    assert metrics["auroc"] == 1 and metrics["auprc"] == 1 and metrics["true_positive"] == 4
    assert .0 <= select_threshold(frame, "youden") <= 1
    calibration = calibration_metrics(frame)
    assert calibration["brier_score"] >= 0 and calibration["expected_calibration_error"] >= 0
    with pytest.raises(ValueError, match="absolute"):
        validate_predictions(frame.assign(image_path="C:/private/image.png"))
    with pytest.raises(ValueError, match="validation"):
        select_threshold(_predictions("test"))


def test_temperature_bootstrap_ensemble_uncertainty_and_selective_prediction() -> None:
    frame = _predictions()
    temperature = fit_temperature(frame)
    assert temperature["temperature"] > 0
    first = bootstrap_confidence_intervals(frame, .5, iterations=20, seed=9)
    second = bootstrap_confidence_intervals(frame, .5, iterations=20, seed=9)
    pd.testing.assert_frame_equal(first, second)
    ensemble = aggregate_ensemble([frame, frame.assign(logit=frame.logit + .1)])
    assert {"mutual_information", "probability_variance", "logit_variance"} <= set(ensemble)
    with pytest.raises(ValueError, match="mismatched"):
        aggregate_ensemble([frame, frame.iloc[::-1].reset_index(drop=True)])
    selective, curve = selective_prediction(add_deterministic_uncertainty(frame), .5)
    assert selective.retained_sample_count.is_monotonic_decreasing
    assert curve.coverage.is_monotonic_increasing


def test_bootstrap_returns_summary_when_every_resample_fails() -> None:
    frame = _predictions().assign(binary_target=0, original_label=0)

    summary = bootstrap_confidence_intervals(frame, .5, iterations=8, seed=123)

    assert summary["metric"].tolist() == [
        "auroc", "auprc", "sensitivity", "specificity", "brier_score",
        "expected_calibration_error",
    ]
    assert summary["lower_95"].isna().all()
    assert summary["upper_95"].isna().all()
    assert (summary["valid_iterations"] == 0).all()
    assert (summary["failed_iterations"] == 8).all()
    assert (summary["iterations"] == 8).all()
    assert (summary["seed"] == 123).all()
    assert "sampled bootstrap frame contains only one target class" in summary[
        "failure_reasons"
    ].iloc[0]


def test_bootstrap_allows_repeated_patient_samples_with_both_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RepeatedPatientGenerator:
        def choice(
            self,
            patients: np.ndarray,
            size: int,
            replace: bool,
        ) -> np.ndarray:
            assert size == len(patients)
            assert replace
            return np.array(["p0", "p1", "p1", "p2"])

    generator = RepeatedPatientGenerator()
    monkeypatch.setattr(evaluation_core.np.random, "default_rng", lambda seed: generator)

    summary = bootstrap_confidence_intervals(_predictions(), .5, iterations=1, seed=7)

    assert (summary["valid_iterations"] == 1).all()
    assert (summary["failed_iterations"] == 0).all()


def test_bootstrap_two_class_predictions_have_successful_iterations() -> None:
    summary = bootstrap_confidence_intervals(_predictions(), .5, iterations=100, seed=17)

    assert (summary["valid_iterations"] > 0).all()


def test_temperature_improves_miscalibrated_synthetic_predictions() -> None:
    logits = np.repeat([-2.0, 2.0], 40)
    targets = np.array([0] * 30 + [1] * 10 + [0] * 10 + [1] * 30)
    probabilities = 1 / (1 + np.exp(-logits))
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(80)], "study_id": [f"s{i}" for i in range(80)],
        "image_path": [f"images/{i}.png" for i in range(80)], "original_label": targets,
        "binary_target": targets, "logit": logits, "probability": probabilities,
        "predicted_class": (probabilities >= .5).astype(int), "split": "validation",
        "model_name": "toy", "run_id": "run"})
    temperature = fit_temperature(frame)["temperature"]
    assert calibration_metrics(apply_temperature(frame, temperature))["brier_score"] < calibration_metrics(frame)["brier_score"]
    assert failure_detection(frame, .5)["error_count"] > 0


def test_master_evaluation_generates_portable_figures(tmp_path: Path) -> None:
    validation = _predictions()
    test = _predictions("test")
    evaluate_predictions(validation, test, tmp_path, "fixed_0.5", 10, 7)
    for filename in ("metrics.json", "metrics.csv", "calibration.json", "bootstrap_confidence_intervals.csv", "selective_prediction.csv", "failure_detection.json", "risk_coverage.csv", "reliability_diagram.png", "roc_curve.png", "pr_curve.png", "risk_coverage.png", "uncertainty_distribution.png"):
        assert (tmp_path / filename).is_file()


def test_deep_ensemble_comparison_and_paired_statistics(tmp_path: Path) -> None:
    single = _predictions()
    ensemble = single.assign(
        probability=[.02, .08, .15, .3, .7, .85, .92, .98],
        logit=np.log(np.array([.02, .08, .15, .3, .7, .85, .92, .98]) / np.array([.98, .92, .85, .7, .3, .15, .08, .02])),
    )

    result = compare_single_and_ensemble(single, ensemble, tmp_path, bootstrap_iterations=20)

    assert "p_value" in delong_auroc_test(single, ensemble)
    assert result["paired_bootstrap_auroc"]["iterations"] == 20
    for filename in ("single_confidence_histogram.png", "ensemble_confidence_histogram.png", "single_vs_ensemble.json"):
        assert (tmp_path / filename).is_file()
