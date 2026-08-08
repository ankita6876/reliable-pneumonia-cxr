"""Leakage-safe evaluation utilities for binary pneumonia classifiers."""

from .core import (
    PREDICTION_COLUMNS,
    aggregate_ensemble,
    align_paired_external_predictions,
    bootstrap_confidence_intervals,
    calibration_metrics,
    compare_single_and_ensemble,
    delong_auroc_test,
    evaluate_predictions,
    external_threshold_metadata,
    failure_detection_all,
    failure_detection_table,
    fit_temperature,
    roc_threshold_analysis,
    select_threshold,
    validate_predictions,
)

__all__ = [
    "PREDICTION_COLUMNS",
    "aggregate_ensemble",
    "align_paired_external_predictions",
    "bootstrap_confidence_intervals",
    "calibration_metrics",
    "compare_single_and_ensemble",
    "delong_auroc_test",
    "evaluate_predictions",
    "external_threshold_metadata",
    "failure_detection_all",
    "failure_detection_table",
    "fit_temperature",
    "roc_threshold_analysis",
    "select_threshold",
    "validate_predictions",
]
