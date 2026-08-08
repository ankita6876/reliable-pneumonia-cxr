"""Evaluate RSNA using threshold and temperature frozen from CheXpert validation."""
# ruff: noqa: E501, E701, E702
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.evaluation.core import (  # noqa: E402
    apply_temperature, bootstrap_confidence_intervals, calibration_metrics, discrimination_metrics,
    external_threshold_metadata,
    failure_detection_all, failure_detection_table, fit_temperature, selective_prediction_all,
    select_threshold, validate_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-predictions", required=True); parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--dataset-name", default="External")
    parser.add_argument("--threshold-method", choices=("youden", "max_f1", "fixed_0.5"), default="youden")
    parser.add_argument("--frozen-threshold", type=float, help="Already selected from non-external validation data; never optimized on RSNA.")
    parser.add_argument("--threshold-source", help="Required provenance source for --frozen-threshold.")
    parser.add_argument("--threshold-selection-dataset", default="CheXpert validation", help="Dataset that selected the supplied threshold; external datasets are rejected.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000); parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _save_figures(frame: pd.DataFrame, threshold: float, curve: pd.DataFrame, failure: pd.DataFrame, output: Path, dataset_name: str = "External") -> None:
    y, p = frame.binary_target.to_numpy(int), frame.probability.to_numpy(float)
    fpr, tpr, _ = roc_curve(y, p); fig, ax = plt.subplots(); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{dataset_name} ROC curve"); fig.tight_layout(); fig.savefig(output / "external_roc_curve.png", dpi=300); plt.close(fig)
    recall, precision, _ = precision_recall_curve(y, p); fig, ax = plt.subplots(); ax.plot(recall, precision); ax.set(xlabel="Recall", ylabel="Precision", title=f"{dataset_name} precision-recall curve"); fig.tight_layout(); fig.savefig(output / "external_precision_recall_curve.png", dpi=300); plt.close(fig)
    observed, predicted = calibration_curve(y, p, n_bins=10); fig, ax = plt.subplots(); ax.plot(predicted, observed, "o-"); ax.plot([0, 1], [0, 1], "--"); ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"{dataset_name} reliability diagram"); fig.tight_layout(); fig.savefig(output / "external_reliability_diagram.png", dpi=300); plt.close(fig)
    matrix = confusion_matrix(y, p >= threshold, labels=[0, 1]); fig, ax = plt.subplots(); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax); ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Target", title="RSNA confusion matrix"); [ax.text(j, i, value, ha="center", va="center") for i, row in enumerate(matrix) for j, value in enumerate(row)]; fig.tight_layout(); fig.savefig(output / "external_confusion_matrix.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots()
    for name, item in curve.groupby("uncertainty_measure"): ax.plot(item.coverage, item.risk, label=name)
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", title="RSNA risk-coverage"); ax.legend(fontsize="small"); fig.tight_layout(); fig.savefig(output / "external_risk_coverage_curve.png", dpi=300); plt.close(fig)
    errors = (p >= threshold) != y; fig, ax = plt.subplots(); ax.hist(frame.loc[~errors, "uncertainty"], alpha=.6, label="Correct"); ax.hist(frame.loc[errors, "uncertainty"], alpha=.6, label="Incorrect"); ax.set(xlabel="Uncertainty", ylabel="Patient count", title="RSNA uncertainty by outcome"); ax.legend(); fig.tight_layout(); fig.savefig(output / "external_uncertainty_by_outcome.png", dpi=300); plt.close(fig)
    valid = failure.dropna(subset=["error_detection_auroc"]); fig, ax = plt.subplots(); ax.barh(valid.uncertainty_measure, valid.error_detection_auroc); ax.set(xlabel="Error-detection AUROC", title="RSNA uncertainty failure detection"); fig.tight_layout(); fig.savefig(output / "external_uncertainty_failure_detection.png", dpi=300); plt.close(fig)


def main() -> None:
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    validation = validate_predictions(pd.read_csv(args.validation_predictions))
    external = validate_predictions(pd.read_csv(args.external_predictions))
    if set(validation.split) != {"validation"}: raise ValueError("Validation predictions must contain only CheXpert validation rows.")
    if set(external.split) != {"external_test"}: raise ValueError("External predictions must contain only external_test rows.")
    if args.frozen_threshold is None:
        threshold = select_threshold(validation, args.threshold_method)
        threshold_metadata = external_threshold_metadata(
            threshold=threshold, source=str(Path(args.validation_predictions).resolve()),
            method=args.threshold_method, selection_dataset="CheXpert validation",
        )
    else:
        if not args.threshold_source:
            raise ValueError("--threshold-source is required with --frozen-threshold.")
        threshold_metadata = external_threshold_metadata(
            threshold=args.frozen_threshold, source=args.threshold_source,
            method="supplied_frozen_validation_threshold",
            selection_dataset=args.threshold_selection_dataset,
        )
        threshold = float(threshold_metadata["threshold"])
    temperature = fit_temperature(validation)
    external = apply_temperature(external, float(temperature["temperature"]))
    metrics = {**discrimination_metrics(external, threshold), **calibration_metrics(external), **threshold_metadata, "temperature": temperature["temperature"], "parameter_source": "CheXpert validation"}
    (output / "external_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True)); pd.DataFrame([metrics]).drop(columns=["reliability"]).to_csv(output / "external_metrics.csv", index=False)
    bootstrap_confidence_intervals(external, threshold, args.bootstrap_iterations, args.seed).to_csv(output / "external_bootstrap_confidence_intervals.csv", index=False)
    selective, curve = selective_prediction_all(external, threshold); selective.to_csv(output / "external_selective_prediction.csv", index=False)
    failure = failure_detection_table(failure_detection_all(external, threshold)); failure.to_csv(output / "external_failure_detection.csv", index=False)
    _save_figures(external, threshold, curve, failure, output, args.dataset_name)


if __name__ == "__main__":
    main()
