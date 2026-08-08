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
    external_threshold_metadata,
    evaluate_predictions,
    failure_detection,
    fit_temperature,
    failure_detection_all,
    failure_detection_table,
    selective_prediction,
    selective_prediction_all,
    select_threshold,
    metrics_at_threshold,
    roc_threshold_analysis,
    validate_predictions,
)


def _predictions(split: str = "validation") -> pd.DataFrame:
    probabilities = np.array([.05, .15, .25, .4, .6, .75, .85, .95])
    return pd.DataFrame({"patient_id":[f"p{i//2}" for i in range(8)], "study_id":[f"s{i}" for i in range(8)], "image_path":[f"images/{i}.png" for i in range(8)], "original_label":[0,0,0,0,1,1,1,1], "binary_target":[0,0,0,0,1,1,1,1], "logit":np.log(probabilities/(1-probabilities)), "probability":probabilities, "predicted_class":(probabilities>=.5).astype(int), "split":split, "model_name":"toy", "run_id":"run"})


def _patient_balanced_predictions(
    split: str,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    targets = np.tile([0, 1], 4)
    return pd.DataFrame(
        {
            "patient_id": [f"p{i // 2}" for i in range(8)],
            "study_id": [f"s{i}" for i in range(8)],
            "image_path": [f"images/{i}.png" for i in range(8)],
            "original_label": targets,
            "binary_target": targets,
            "logit": np.log(probabilities / (1 - probabilities)),
            "probability": probabilities,
            "predicted_class": (probabilities >= .5).astype(int),
            "split": split,
            "model_name": "toy",
            "run_id": "run",
        }
    )


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


def test_youden_j_returns_full_roc_and_highest_threshold_on_tie() -> None:
    result = roc_threshold_analysis(
        np.array([0, 0, 1, 1]), np.array([.1, .4, .4, .9]),
    )
    assert {"fpr", "tpr", "thresholds", "youden_j", "selected_threshold", "selected_youden_j"} <= set(result)
    assert result["selected_threshold"] == pytest.approx(.9)
    assert result["selected_youden_j"] == pytest.approx(.5)
    assert "highest finite threshold" in result["tie_breaking"]


@pytest.mark.parametrize(
    ("targets", "probabilities", "message"),
    [
        (np.array([], dtype=int), np.array([], dtype=float), "non-empty"),
        (np.array([0, 0]), np.array([.1, .2]), "both target classes"),
        (np.array([0, 1]), np.array([.1, np.nan]), "NaN or infinite"),
    ],
)
def test_youden_j_rejects_invalid_input(targets, probabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        roc_threshold_analysis(targets, probabilities)


def test_supplied_external_threshold_requires_non_external_provenance() -> None:
    metadata = external_threshold_metadata(
        threshold=.37, source="configs/operating_thresholds/original_seed_42.json",
        method="supplied_frozen_validation_threshold", selection_dataset="CheXpert validation",
    )
    assert metadata["threshold"] == pytest.approx(.37)
    assert metadata["threshold_selection_dataset"] == "CheXpert validation"
    with pytest.raises(ValueError, match="must not select"):
        external_threshold_metadata(
            threshold=.37, source="RSNA", method="youden_j_roc", selection_dataset="RSNA external",
        )


def test_supplied_threshold_application_is_threshold_only() -> None:
    frame = _predictions()
    metrics = discrimination_metrics(frame, .75)
    assert metrics["threshold"] == pytest.approx(.75)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)


def test_temperature_bootstrap_ensemble_uncertainty_and_selective_prediction() -> None:
    frame = _predictions()
    temperature = fit_temperature(frame)
    assert temperature["temperature"] > 0
    first = bootstrap_confidence_intervals(frame, .5, iterations=20, seed=9)
    second = bootstrap_confidence_intervals(frame, .5, iterations=20, seed=9)
    pd.testing.assert_frame_equal(first, second)
    ensemble = aggregate_ensemble([frame, frame.assign(logit=frame.logit + .1)])
    assert {"mutual_information", "probability_variance", "logit_variance"} <= set(ensemble)
    reordered = aggregate_ensemble([frame, frame.iloc[::-1].reset_index(drop=True)])
    assert reordered["patient_id"].tolist() == sorted(frame["patient_id"].tolist())
    selective, curve = selective_prediction(add_deterministic_uncertainty(frame), .5)
    assert selective.retained_sample_count.is_monotonic_decreasing
    assert curve.coverage.is_monotonic_increasing


def test_selective_prediction_uses_deterministic_rank_retention_for_tiny_tied_data() -> None:
    frame = _predictions().iloc[:3].copy()
    frame["uncertainty"] = [.2, .2, .2]
    frame["confidence"] = .8
    frame["confidence_uncertainty"] = .2
    frame["predictive_entropy"] = .5
    selective, _ = selective_prediction(frame, .5, coverages=(.01, .5, 1.0))

    assert selective["retained_sample_count"].tolist() == [1, 2, 3]
    assert selective["coverage"].tolist() == [1 / 3, 2 / 3, 1.0]
    assert selective["requested_coverage"].tolist() == [.01, .5, 1.0]


def test_selective_prediction_explicit_minimum_cutoff_returns_structured_empty_row() -> None:
    frame = _predictions().iloc[:2].copy()
    frame["uncertainty"] = [.2, .3]
    frame["confidence"] = [.8, .7]
    frame["confidence_uncertainty"] = [.2, .3]
    frame["predictive_entropy"] = [.5, .6]
    selective, _ = selective_prediction(frame, .5, coverages=(.5,), cutoffs={.5: .1})

    row = selective.iloc[0]
    assert row.retained_sample_count == 0 and row.coverage == 0
    assert np.isnan(row.accuracy) and np.isnan(row.risk)


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
    for filename in (
        "metrics.json",
        "metrics.csv",
        "calibration.json",
        "bootstrap_confidence_intervals.csv",
        "selective_prediction.csv",
        "failure_detection.json",
        "risk_coverage.csv",
        "reliability_diagram.png",
        "roc_curve.png",
        "pr_curve.png",
        "risk_coverage.png",
        "uncertainty_distribution.png",
        "validation_uncertainty_auroc_comparison.png",
        "validation_uncertainty_auprc_comparison.png",
        "validation_uncertainty_risk_coverage.png",
    ):
        assert (tmp_path / filename).is_file()


def test_evaluation_writes_split_specific_patient_bootstrap_intervals(
    tmp_path: Path,
) -> None:
    validation = _patient_balanced_predictions(
        "validation",
        np.array([.05, .95, .1, .9, .2, .8, .25, .75]),
    )
    test = _patient_balanced_predictions(
        "test",
        np.array([.35, .65, .4, .6, .45, .55, .3, .7]),
    )

    metrics = evaluate_predictions(
        validation,
        test,
        tmp_path,
        "fixed_0.5",
        20,
        11,
    )

    validation_bootstrap = pd.read_csv(
        tmp_path / "validation_bootstrap_confidence_intervals.csv"
    )
    test_bootstrap = pd.read_csv(tmp_path / "test_bootstrap_confidence_intervals.csv")
    legacy_bootstrap = pd.read_csv(tmp_path / "bootstrap_confidence_intervals.csv")
    temperature = fit_temperature(validation)["temperature"]
    calibrated_test = apply_temperature(test, temperature)
    expected_test_brier = calibration_metrics(calibrated_test)["brier_score"]
    expected_test_sensitivity = metrics_at_threshold(
        calibrated_test["binary_target"].to_numpy(int),
        calibrated_test["probability"].to_numpy(float),
        .5,
    )["sensitivity"]

    assert validation_bootstrap["metric"].tolist() == [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "brier_score",
        "expected_calibration_error",
    ]
    assert test_bootstrap["metric"].tolist() == validation_bootstrap["metric"].tolist()
    assert {"point_estimate", "lower_95", "upper_95", "valid_iterations", "failed_iterations", "iterations", "seed", "failure_reasons"} <= set(test_bootstrap.columns)
    assert (validation_bootstrap["valid_iterations"] == 20).all()
    assert (test_bootstrap["valid_iterations"] == 20).all()
    assert (validation_bootstrap["failed_iterations"] == 0).all()
    assert (test_bootstrap["failed_iterations"] == 0).all()
    pd.testing.assert_frame_equal(validation_bootstrap, legacy_bootstrap)
    assert test_bootstrap.loc[test_bootstrap["metric"] == "brier_score", "point_estimate"].item() == pytest.approx(expected_test_brier)
    assert test_bootstrap.loc[test_bootstrap["metric"] == "sensitivity", "point_estimate"].item() == pytest.approx(expected_test_sensitivity)
    assert metrics["validation"]["temperature"] == pytest.approx(temperature)
    assert metrics["test"]["temperature"] == pytest.approx(temperature)


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


def test_disagreement_uncertainty_detects_synthetic_errors_better_than_confidence() -> None:
    targets = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    member_probabilities = np.array([
        [.6, .4, .6, .4, .95, .05, .95, .05],
        [.6, .4, .6, .4, .15, .85, .15, .85],
        [.6, .4, .6, .4, .7, .3, .7, .3],
    ])
    members = []
    for probabilities in member_probabilities:
        member = _predictions().assign(
            binary_target=targets,
            original_label=targets,
            probability=probabilities,
            logit=np.log(probabilities / (1 - probabilities)),
        )
        members.append(member)

    ensemble = aggregate_ensemble(members)
    analyses = failure_detection_all(ensemble, threshold=.5)
    table = failure_detection_table(analyses)
    selective, curves = selective_prediction_all(ensemble, threshold=.5)

    assert analyses["binary_member_disagreement_rate"]["error_detection_auroc"] > analyses["confidence_uncertainty"]["error_detection_auroc"]
    assert analyses["member_probability_variance"]["error_detection_auroc"] > analyses["confidence_uncertainty"]["error_detection_auroc"]
    assert set(table["uncertainty_measure"]) >= {
        "binary_member_disagreement_rate",
        "confidence_uncertainty",
    }
    assert set(selective["uncertainty_measure"]) == set(curves["uncertainty_measure"])


def test_identical_ensemble_members_have_negligible_epistemic_uncertainty() -> None:
    frame = _predictions()
    ensemble = aggregate_ensemble([frame, frame.copy(), frame.copy()])

    assert np.allclose(ensemble["mutual_information"], 0, atol=1e-12)
    assert np.allclose(ensemble["member_probability_variance"], 0, atol=1e-15)
    assert np.allclose(ensemble["member_probability_standard_deviation"], 0, atol=1e-15)
    assert {"member_probability_0", "member_probability_1", "member_probability_2"} <= set(ensemble)
