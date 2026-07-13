"""Statistical evaluation that never learns parameters from held-out test data."""
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, roc_curve,
)

PREDICTION_COLUMNS = (
    "patient_id", "study_id", "image_path", "original_label", "binary_target", "logit",
    "probability", "predicted_class", "split", "model_name", "run_id",
)
IDENTITY_COLUMNS = ("patient_id", "study_id", "image_path", "split")
COVERAGES = (1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)


def validate_predictions(frame: pd.DataFrame, *, require_logits: bool = True) -> pd.DataFrame:
    """Validate schema, finite values, binary labels, portable paths, and unique identities."""
    required = set(PREDICTION_COLUMNS if require_logits else ("patient_id", "study_id", "image_path", "binary_target", "probability", "split"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing required column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Prediction CSV contains no rows.")
    if frame[list(IDENTITY_COLUMNS)].isna().any().any() or frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Prediction sample identities must be present and unique.")
    if frame["image_path"].map(_is_absolute_path).any():
        raise ValueError("Prediction CSV must not contain absolute image paths.")
    targets = pd.to_numeric(frame["binary_target"], errors="coerce")
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ValueError("binary_target must contain only 0 and 1.")
    for name in ("probability", "logit") if require_logits else ("probability",):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite numeric values.")
    probabilities = frame["probability"].to_numpy(float)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probability must be in [0, 1].")
    return frame.copy()


def select_threshold(frame: pd.DataFrame, method: str = "youden") -> float:
    """Select an operating threshold from validation predictions only."""
    frame = validate_predictions(frame)
    if frame["split"].nunique() != 1 or frame["split"].iloc[0] != "validation":
        raise ValueError("Operating thresholds may only be selected from validation predictions.")
    y, p = _arrays(frame)
    if method == "fixed_0.5":
        return 0.5
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both target classes.")
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        return float(np.clip(thresholds[np.argmax(tpr - fpr)], 0, 1))
    if method == "max_f1":
        thresholds = np.unique(p)
        scores = [metrics_at_threshold(y, p, float(t))["f1"] for t in thresholds]
        return float(thresholds[int(np.argmax(scores))])
    raise ValueError("threshold method must be youden, max_f1, or fixed_0.5.")


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return discrimination-independent operating-point metrics and counts."""
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"threshold": threshold, "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "sensitivity": _safe_divide(tp, tp + fn), "specificity": _safe_divide(tn, tn + fp),
            "precision": _safe_divide(tp, tp + fp), "negative_predictive_value": _safe_divide(tn, tn + fn),
            "f1": _safe_divide(2 * tp, 2 * tp + fp + fn), "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp)}


def discrimination_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    """Calculate discrimination plus an operating point without any optimization."""
    frame = validate_predictions(frame)
    y, p = _arrays(frame)
    if len(np.unique(y)) != 2:
        raise ValueError("AUROC and AUPRC require both target classes.")
    result: dict[str, float | int] = {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    result.update(metrics_at_threshold(y, p, threshold))
    return result


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10, adaptive: bool = False) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)) if adaptive else np.linspace(0, 1, bins + 1)
    edges[0], edges[-1] = 0.0, 1.0
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if mask.any(): total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return proper scores, ECE variants, logistic calibration fit, and reliability data."""
    frame = validate_predictions(frame)
    y, p = _arrays(frame)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    try:
        from sklearn.linear_model import LogisticRegression
        fit = LogisticRegression(C=1e6, solver="lbfgs").fit(logits.reshape(-1, 1), y)
        intercept, slope = float(fit.intercept_[0]), float(fit.coef_[0, 0])
    except ValueError:
        intercept, slope = float("nan"), float("nan")
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    return {"negative_log_likelihood": float(log_loss(y, clipped, labels=[0, 1])), "brier_score": float(brier_score_loss(y, p)),
            "expected_calibration_error": expected_calibration_error(y, p), "adaptive_calibration_error": expected_calibration_error(y, p, adaptive=True),
            "calibration_intercept": intercept, "calibration_slope": slope,
            "reliability": [{"mean_predicted_probability": float(x), "fraction_positive": float(z)} for x, z in zip(predicted, observed)]}


def fit_temperature(validation: pd.DataFrame) -> dict[str, float | int]:
    """Fit one positive temperature exclusively on validation logits using NLL."""
    validation = validate_predictions(validation)
    if validation["split"].nunique() != 1 or validation["split"].iloc[0] != "validation":
        raise ValueError("Temperature scaling may only be fitted from validation predictions.")
    logits = torch.tensor(validation["logit"].to_numpy(float), dtype=torch.float64)
    targets = torch.tensor(validation["binary_target"].to_numpy(float), dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], lr=0.25, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    def closure() -> torch.Tensor:
        optimizer.zero_grad(); loss = loss_fn(logits / parameter.exp(), targets); loss.backward(); return loss
    optimizer.step(closure)
    temperature = float(parameter.detach().exp().clamp_min(1e-6))
    return {"temperature": temperature, "fit_split": "validation", "objective": "negative_log_likelihood", "sample_count": len(validation)}


def apply_temperature(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    if not np.isfinite(temperature) or temperature <= 0: raise ValueError("temperature must be positive and finite.")
    result = validate_predictions(frame); result["logit"] /= temperature; result["probability"] = _sigmoid(result["logit"].to_numpy(float)); return result


def add_deterministic_uncertainty(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_predictions(frame); p = result["probability"].to_numpy(float)
    result["confidence"] = np.maximum(p, 1 - p); result["uncertainty"] = 1 - result["confidence"]
    result["predictive_entropy"] = _entropy(p); return result


def aggregate_ensemble(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Average aligned model outputs and calculate decomposition uncertainty measures."""
    members = [validate_predictions(item).reset_index(drop=True) for item in frames]
    if len(members) < 2: raise ValueError("An ensemble requires at least two prediction CSV files.")
    identity = members[0][list(IDENTITY_COLUMNS)]
    for member in members[1:]:
        if not identity.equals(member[list(IDENTITY_COLUMNS)]): raise ValueError("Ensemble inputs have mismatched sample identities or ordering.")
        if not np.array_equal(members[0]["binary_target"], member["binary_target"]): raise ValueError("Ensemble inputs have mismatched binary targets.")
    probabilities = np.stack([item["probability"].to_numpy(float) for item in members]); logits = np.stack([item["logit"].to_numpy(float) for item in members])
    result = members[0].copy(); result["logit"] = logits.mean(axis=0); result["probability"] = probabilities.mean(axis=0); result["predicted_class"] = (result["probability"] >= .5).astype(int)
    result["ensemble_mean_probability"] = result["probability"]; result["predictive_entropy"] = _entropy(result["probability"].to_numpy(float)); result["expected_entropy"] = _entropy(probabilities).mean(axis=0); result["mutual_information"] = result["predictive_entropy"] - result["expected_entropy"]; result["probability_variance"] = probabilities.var(axis=0); result["logit_variance"] = logits.var(axis=0)
    return result


def delong_auroc_test(single: pd.DataFrame, ensemble: pd.DataFrame) -> dict[str, float]:
    """Paired DeLong test for two AUROCs on identical binary-labelled samples."""
    _validate_paired_predictions(single, ensemble)
    y, first = _arrays(single)
    _, second = _arrays(ensemble)
    positives = y == 1
    negatives = y == 0
    first_positive, first_negative = first[positives], first[negatives]
    second_positive, second_negative = second[positives], second[negatives]
    first_v10, first_v01 = _delong_placements(first_positive, first_negative)
    second_v10, second_v01 = _delong_placements(second_positive, second_negative)
    covariance = np.cov(np.vstack((first_v10, second_v10)), ddof=1) / len(first_v10)
    covariance += np.cov(np.vstack((first_v01, second_v01)), ddof=1) / len(first_v01)
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    difference = float(roc_auc_score(y, second) - roc_auc_score(y, first))
    if variance <= 0: return {"auroc_difference": difference, "z_score": float("nan"), "p_value": float("nan")}
    z_score = difference / np.sqrt(variance)
    from scipy.stats import norm
    return {"auroc_difference": difference, "z_score": float(z_score), "p_value": float(2 * norm.sf(abs(z_score)))}


def paired_bootstrap_auroc(single: pd.DataFrame, ensemble: pd.DataFrame, iterations: int = 1000, seed: int = 42) -> dict[str, float | int]:
    """Patient-level paired bootstrap confidence interval for ensemble-minus-single AUROC."""
    _validate_paired_predictions(single, ensemble)
    patients = single["patient_id"].unique(); rng = np.random.default_rng(seed); differences = []
    for _ in range(iterations):
        selected = rng.choice(patients, len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(single["patient_id"].to_numpy() == patient) for patient in selected])
        y = single.iloc[indices]["binary_target"].to_numpy(int)
        if len(np.unique(y)) == 2:
            differences.append(float(roc_auc_score(y, ensemble.iloc[indices]["probability"]) - roc_auc_score(y, single.iloc[indices]["probability"])))
    values = np.asarray(differences)
    return {"difference": float(roc_auc_score(ensemble["binary_target"], ensemble["probability"]) - roc_auc_score(single["binary_target"], single["probability"])), "lower_95": float(np.quantile(values, .025)) if len(values) else float("nan"), "upper_95": float(np.quantile(values, .975)) if len(values) else float("nan"), "valid_iterations": len(values), "failed_iterations": iterations - len(values), "iterations": iterations, "seed": seed}


def compare_single_and_ensemble(single: pd.DataFrame, ensemble: pd.DataFrame, output_dir: Path | str, threshold: float = .5, bootstrap_iterations: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Write matched single-model versus ensemble metrics and comparison figures."""
    _validate_paired_predictions(single, ensemble)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    results = {"single_model": {**discrimination_metrics(single, threshold), **calibration_metrics(single), **failure_detection(single, threshold)}, "deep_ensemble": {**discrimination_metrics(ensemble, threshold), **calibration_metrics(ensemble), **failure_detection(ensemble, threshold)}, "delong": delong_auroc_test(single, ensemble), "paired_bootstrap_auroc": paired_bootstrap_auroc(single, ensemble, bootstrap_iterations, seed)}
    (output / "single_vs_ensemble.json").write_text(json.dumps(results, indent=2, allow_nan=True))
    for label, frame in (("single", single), ("ensemble", ensemble)):
        _, curve = selective_prediction(frame, threshold); _plots(frame, curve, output, label); _confidence_plot(frame, output, label)
    return results


def bootstrap_confidence_intervals(frame: pd.DataFrame, threshold: float, iterations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Patient-level percentile bootstrap, recording failed one-class resamples explicitly."""
    frame = validate_predictions(frame); y, p = _arrays(frame); patients = frame["patient_id"].unique(); rng = np.random.default_rng(seed); rows = []
    for iteration in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True); indices = np.concatenate([np.flatnonzero(frame["patient_id"].to_numpy() == patient) for patient in selected])
        try:
            metrics = discrimination_metrics(frame.iloc[indices], threshold); metrics.update(calibration_metrics(frame.iloc[indices])); rows.append({"iteration": iteration, "status": "valid", **{key: metrics[key] for key in ("auroc", "auprc", "sensitivity", "specificity", "brier_score", "expected_calibration_error")}})
        except ValueError as error: rows.append({"iteration": iteration, "status": "failed", "reason": str(error)})
    values = pd.DataFrame(rows)
    valid = values.loc[values["status"] == "valid"]
    summary = []
    for metric in ("auroc", "auprc", "sensitivity", "specificity", "brier_score", "expected_calibration_error"):
        series = valid[metric].dropna() if metric in valid else pd.Series(dtype=float)
        summary.append({"metric": metric, "lower_95": series.quantile(.025) if len(series) else np.nan, "upper_95": series.quantile(.975) if len(series) else np.nan, "valid_iterations": len(valid), "failed_iterations": len(values) - len(valid), "iterations": iterations, "seed": seed})
    return pd.DataFrame(summary)


def failure_detection(frame: pd.DataFrame, threshold: float, uncertainty_column: str = "uncertainty") -> dict[str, Any]:
    frame = add_deterministic_uncertainty(frame) if uncertainty_column not in frame else frame.copy(); validate_predictions(frame)
    y, p = _arrays(frame); error = ((p >= threshold).astype(int) != y).astype(int); u = frame[uncertainty_column].to_numpy(float)
    result: dict[str, Any] = {"error_count": int(error.sum()), "correct_count": int((1-error).sum())}
    if len(np.unique(error)) == 2: result.update(error_detection_auroc=float(roc_auc_score(error,u)), error_detection_auprc=float(average_precision_score(error,u)))
    else: result.update(error_detection_auroc=None, error_detection_auprc=None)
    result["uncertainty_correct"] = _distribution(u[error == 0]); result["uncertainty_incorrect"] = _distribution(u[error == 1])
    groups = {"tp": (y==1)&(error==0), "tn": (y==0)&(error==0), "fp": (y==0)&(error==1), "fn": (y==1)&(error==1)}; result["by_outcome"] = {key:_distribution(u[mask]) for key,mask in groups.items()}
    if error.sum() and (1-error).sum():
        from scipy.stats import mannwhitneyu
        statistic, pvalue = mannwhitneyu(u[error==1], u[error==0], alternative="two-sided"); result["mann_whitney_u"] = float(statistic); result["mann_whitney_pvalue"] = float(pvalue); result["rank_biserial_effect_size"] = float(2*statistic/(error.sum()*(1-error).sum())-1)
    return result


def selective_prediction(frame: pd.DataFrame, threshold: float, coverages: tuple[float,...] = COVERAGES, cutoffs: dict[float,float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_deterministic_uncertainty(frame) if "uncertainty" not in frame else frame.copy(); validate_predictions(frame); y,p=_arrays(frame); rows=[]; curve=[]
    for coverage in coverages:
        cutoff = cutoffs[coverage] if cutoffs else float(np.quantile(frame.uncertainty, coverage, method="higher")); retained=frame.loc[frame.uncertainty<=cutoff]; metrics=discrimination_metrics(retained,threshold) if len(np.unique(retained.binary_target))==2 else metrics_at_threshold(retained.binary_target.to_numpy(int),retained.probability.to_numpy(float),threshold)
        metrics.update(coverage=coverage, uncertainty_cutoff=cutoff, retained_sample_count=len(retained), deferred_sample_count=len(frame)-len(retained), risk=1-metrics["accuracy"]); rows.append(metrics)
    ordered=frame.sort_values("uncertainty").reset_index(drop=True)
    for count in range(1,len(ordered)+1): curve.append({"coverage":count/len(ordered),"risk":float(np.mean((ordered.probability.iloc[:count].to_numpy()>=threshold)!=ordered.binary_target.iloc[:count].to_numpy()))})
    return pd.DataFrame(rows),pd.DataFrame(curve)


def evaluate_predictions(validation: pd.DataFrame, test: pd.DataFrame | None, output_dir: Path | str, threshold_method: str, bootstrap_iterations: int, seed: int) -> dict[str, Any]:
    """Master evaluation; validation selects all parameters and test only receives them frozen."""
    validation=validate_predictions(validation); test=validate_predictions(test) if test is not None else None
    if validation["split"].nunique()!=1 or validation["split"].iloc[0]!="validation": raise ValueError("--validation-predictions must contain only validation rows.")
    if test is not None and (test["split"].nunique()!=1 or test["split"].iloc[0]!="test"): raise ValueError("--test-predictions must contain only test rows.")
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); threshold=select_threshold(validation,threshold_method); temperature=fit_temperature(validation); datasets={"validation":apply_temperature(validation,temperature["temperature"])}
    if test is not None: datasets["test"]=apply_temperature(test,temperature["temperature"])
    all_metrics={}
    for name,frame in datasets.items():
        metrics={**discrimination_metrics(frame,threshold),**calibration_metrics(frame),"threshold_method":threshold_method,"temperature":temperature["temperature"],"split":name}; all_metrics[name]=metrics
        selective, curve=selective_prediction(frame,threshold,cutoffs={coverage:float(np.quantile(datasets["validation"].uncertainty if "uncertainty" in datasets["validation"] else add_deterministic_uncertainty(datasets["validation"]).uncertainty,coverage,method="higher")) for coverage in COVERAGES}); selective.to_csv(output/f"{name}_selective_prediction.csv",index=False); curve.to_csv(output/f"{name}_risk_coverage.csv",index=False); (output/f"{name}_failure_detection.json").write_text(json.dumps(failure_detection(frame,threshold),indent=2,allow_nan=False)); _plots(frame,curve,output,name)
        if name == "validation":
            selective.to_csv(output / "selective_prediction.csv", index=False); curve.to_csv(output / "risk_coverage.csv", index=False)
            (output / "failure_detection.json").write_text(json.dumps(failure_detection(frame, threshold), indent=2, allow_nan=False))
            for suffix in ("reliability_diagram.png", "roc_curve.png", "pr_curve.png", "risk_coverage.png", "uncertainty_distribution.png"):
                (output / f"validation_{suffix}").replace(output / suffix)
    (output/"metrics.json").write_text(json.dumps(all_metrics,indent=2,allow_nan=False)); pd.DataFrame(all_metrics.values()).to_csv(output/"metrics.csv",index=False); (output/"calibration.json").write_text(json.dumps({"temperature_fit":temperature,"metrics":{k:calibration_metrics(v) for k,v in datasets.items()}},indent=2,allow_nan=False)); bootstrap_summary = bootstrap_confidence_intervals(datasets["validation"],threshold,bootstrap_iterations,seed)
    bootstrap_summary.to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    return all_metrics


def _plots(frame: pd.DataFrame, curve: pd.DataFrame, output: Path, prefix: str) -> None:
    y,p=_arrays(frame); plt.figure(); fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.tight_layout();plt.savefig(output/f"{prefix}_roc_curve.png",dpi=300);plt.close()
    plt.figure(); recall,precision,_=precision_recall_curve(y,p);plt.plot(recall,precision);plt.xlabel("Recall");plt.ylabel("Precision");plt.tight_layout();plt.savefig(output/f"{prefix}_pr_curve.png",dpi=300);plt.close()
    plt.figure(); observed,predicted=calibration_curve(y,p,n_bins=10);plt.plot(predicted,observed,marker="o");plt.plot([0,1],[0,1],linestyle="--");plt.xlabel("Mean predicted probability");plt.ylabel("Observed frequency");plt.tight_layout();plt.savefig(output/f"{prefix}_reliability_diagram.png",dpi=300);plt.close()
    plt.figure();plt.plot(curve.coverage,curve.risk);plt.xlabel("Coverage");plt.ylabel("Risk (error rate)");plt.tight_layout();plt.savefig(output/f"{prefix}_risk_coverage.png",dpi=300);plt.close()
    uncertainty=add_deterministic_uncertainty(frame); errors=(uncertainty.probability.to_numpy()>=.5)!=uncertainty.binary_target.to_numpy();plt.figure();plt.hist(uncertainty.loc[~errors,"uncertainty"],alpha=.6,label="Correct");plt.hist(uncertainty.loc[errors,"uncertainty"],alpha=.6,label="Incorrect");plt.xlabel("Uncertainty");plt.ylabel("Sample count");plt.legend();plt.tight_layout();plt.savefig(output/f"{prefix}_uncertainty_distribution.png",dpi=300);plt.close()


def _confidence_plot(frame: pd.DataFrame, output: Path, prefix: str) -> None:
    """Save a 300 DPI confidence histogram without prescribing colours."""
    probabilities = frame["probability"].to_numpy(float)
    confidence = np.maximum(probabilities, 1 - probabilities)
    plt.figure(); plt.hist(confidence); plt.xlabel("Prediction confidence"); plt.ylabel("Sample count"); plt.tight_layout(); plt.savefig(output / f"{prefix}_confidence_histogram.png", dpi=300); plt.close()


def _validate_paired_predictions(single: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    """Ensure comparison samples and targets align exactly before paired statistics."""
    validate_predictions(single); validate_predictions(ensemble)
    if not single[list(IDENTITY_COLUMNS)].reset_index(drop=True).equals(ensemble[list(IDENTITY_COLUMNS)].reset_index(drop=True)):
        raise ValueError("Single-model and ensemble predictions must have matching identities and ordering.")
    if not np.array_equal(single["binary_target"].to_numpy(), ensemble["binary_target"].to_numpy()):
        raise ValueError("Single-model and ensemble predictions must have matching targets.")


def _delong_placements(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comparison = positive[:, None] - negative[None, :]
    kernel = (comparison > 0).astype(float) + .5 * (comparison == 0)
    return kernel.mean(axis=1), kernel.mean(axis=0)

def _arrays(frame: pd.DataFrame)->tuple[np.ndarray,np.ndarray]: return frame.binary_target.to_numpy(int),frame.probability.to_numpy(float)
def _is_absolute_path(value: object) -> bool:
    """Return whether a serialized path would expose an absolute location."""
    return Path(str(value)).is_absolute()


def _safe_divide(numerator: float | int, denominator: float | int) -> float:
    """Return a metric ratio or NaN when the denominator is unavailable."""
    return float(numerator / denominator) if denominator else float("nan")


def _sigmoid(values:np.ndarray)->np.ndarray: return 1/(1+np.exp(-values))
def _entropy(probabilities:np.ndarray)->np.ndarray:
    p=np.clip(probabilities,1e-12,1-1e-12);return -(p*np.log(p)+(1-p)*np.log(1-p))
def _distribution(values:np.ndarray)->dict[str,float|int|None]: return {"count":int(len(values)),"mean":float(np.mean(values)) if len(values) else None,"median":float(np.median(values)) if len(values) else None}
