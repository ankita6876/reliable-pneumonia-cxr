"""Generate leakage-safe Phase 1.1 threshold and ROC artifacts from saved predictions.

Only CheXpert validation files are used to estimate Youden-J thresholds.  RSNA
predictions are read only after those thresholds are frozen and are never
passed to threshold-selection code.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pneumonia_ai.evaluation.core import metrics_at_threshold, roc_threshold_analysis  # noqa: E402

BASELINE_TAG = "v1-thesis-baseline"
BASELINE_COMMIT = "37966623bb490431308394c1e51f46dd5f870d0d"
ARCHIVES = ROOT / "experiments" / "kaggle_gpu" / "archives"
EXTRACTED = ROOT / "experiments" / "kaggle_gpu" / "extracted"
RUNS = {
    ("original", 42): "A4_original_control_results.zip",
    ("original", 123): "A4_original_seed_123_results.zip",
    ("original", 2026): "A4_original_seed_2026_results.zip",
    ("hard_masked", 42): "thesis_optimisation_results/classification_optimisation/A4_regularised_optimisation",
    ("hard_masked", 123): "A4_hard_masked_seed_123_results.zip",
    ("hard_masked", 2026): "A4_hard_masked_seed_2026_results.zip",
}


def _read_validation(source: str) -> tuple[pd.DataFrame, str]:
    if source.endswith(".zip"):
        path = ARCHIVES / source
        with zipfile.ZipFile(path) as bundle:
            names = [name for name in bundle.namelist() if name.endswith("validation_predictions.csv")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected one validation_predictions.csv, found {names}")
            frame = pd.read_csv(io.BytesIO(bundle.read(names[0])))
        provenance = f"{path.relative_to(ROOT).as_posix()}::{names[0]}"
    else:
        path = EXTRACTED / source / "validation_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        provenance = path.relative_to(ROOT).as_posix()
    label = "label" if "label" in frame else "binary_target" if "binary_target" in frame else None
    if label is None or "probability" not in frame:
        raise ValueError(f"{provenance}: validation predictions need label/binary_target and probability columns")
    if "split" in frame and set(frame["split"].astype(str)) != {"validation"}:
        raise ValueError(f"{provenance}: expected only validation rows")
    return pd.DataFrame({"y_true": frame[label], "y_prob": frame["probability"]}), provenance


def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def derive_thresholds(output: Path, overwrite: bool) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], pd.DataFrame]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    generated = datetime.now(timezone.utc).isoformat()
    for (condition, seed), source in RUNS.items():
        frame, provenance = _read_validation(source)
        analysis = roc_threshold_analysis(frame.y_true.to_numpy(), frame.y_prob.to_numpy())
        record = {
            "dataset_used_for_threshold_selection": "CheXpert validation",
            "condition": condition, "seed": seed,
            "model_identity": f"{condition}_seed_{seed}",
            "threshold_method": "youden_j_roc",
            "selected_threshold": analysis["selected_threshold"],
            "youden_j": analysis["selected_youden_j"],
            "validation_auroc": float(roc_auc_score(frame.y_true, frame.y_prob)),
            "validation_pr_auc": float(average_precision_score(frame.y_true, frame.y_prob)),
            "tie_breaking": analysis["tie_breaking"],
            "generation_timestamp_utc": generated,
            "source_prediction_artifact": provenance,
            "baseline_git_tag": BASELINE_TAG, "baseline_git_commit": BASELINE_COMMIT,
            "selection_uses_external_labels": False,
        }
        _write_json(output / f"{condition}_seed_{seed}.json", record, overwrite)
        records[(condition, seed)], frames[(condition, seed)] = record, frame
    summary = pd.DataFrame(records.values()).sort_values(["condition", "seed"])
    path = output / "chexpert_youden_thresholds_summary.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    summary.to_csv(path, index=False)
    return records, frames


def _external_frames() -> dict[str, pd.DataFrame]:
    path = EXTRACTED / "RSNA_external_statistical_analysis" / "RSNA_external_statistical_analysis" / "paired_predictions.csv"
    frame = pd.read_csv(path)
    result = {}
    for condition, suffix in (("original", "original"), ("hard_masked", "masked")):
        result[condition] = pd.DataFrame({"y_true": frame[f"binary_target_{suffix}"], "y_prob": frame[f"probability_{suffix}"]})
    return result


def sensitivity_table(records: dict[tuple[str, int], dict[str, Any]], internal: dict[tuple[str, int], pd.DataFrame], output: Path, overwrite: bool) -> Path:
    rows = []
    for key, frame in internal.items():
        condition, seed = key
        for threshold_label, threshold, source in (("fixed_0.5", .5, "historical fixed threshold"), ("validation_youden_j", records[key]["selected_threshold"], "CheXpert validation, same run (in-sample operating point)")):
            values = metrics_at_threshold(frame.y_true.to_numpy(int), frame.y_prob.to_numpy(float), threshold)
            rows.append({"dataset": "CheXpert validation", "condition": condition, "seed": seed, "threshold_label": threshold_label, "threshold_source": source, **values, "auroc": roc_auc_score(frame.y_true, frame.y_prob), "pr_auc": average_precision_score(frame.y_true, frame.y_prob)})
    for condition, frame in _external_frames().items():
        record = records[(condition, 42)]
        for threshold_label, threshold, source in (("fixed_0.5", .5, "historical fixed threshold"), ("validation_youden_j", record["selected_threshold"], f"CheXpert validation {condition} seed 42")):
            values = metrics_at_threshold(frame.y_true.to_numpy(int), frame.y_prob.to_numpy(float), threshold)
            rows.append({"dataset": "RSNA external", "condition": condition, "seed": 42, "threshold_label": threshold_label, "threshold_source": source, **values, "auroc": roc_auc_score(frame.y_true, frame.y_prob), "pr_auc": average_precision_score(frame.y_true, frame.y_prob)})
    path = output / "threshold_sensitivity_table.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _style(axis: plt.Axes, title: str) -> None:
    axis.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1, label="Random classifier")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.grid(alpha=.25)
    axis.legend(loc="lower right", fontsize=9)


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def roc_figures(internal: dict[tuple[str, int], pd.DataFrame], output: Path, overwrite: bool) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"{name}.{suffix}" for name in ("chexpert_internal_roc_comparison", "rsna_external_roc_comparison", "chexpert_rsna_roc_comparison") for suffix in ("png", "pdf")]
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError("Refusing to overwrite existing ROC artifact(s)")
    colors = {"original": "#4c78a8", "hard_masked": "#f58518"}
    grid = np.linspace(0, 1, 201)
    def plot_internal(axis: plt.Axes) -> None:
        for condition in ("original", "hard_masked"):
            curves, aucs = [], []
            for seed in (42, 123, 2026):
                frame = internal[(condition, seed)]
                fpr, tpr, _ = roc_curve(frame.y_true, frame.y_prob)
                curves.append(np.interp(grid, fpr, tpr))
                aucs.append(roc_auc_score(frame.y_true, frame.y_prob))
            curves = np.asarray(curves)
            average = curves.mean(axis=0)
            spread = curves.std(axis=0, ddof=1)
            axis.plot(grid, average, color=colors[condition], linewidth=2, label=f"{condition.replace('_', ' ')} (AUROC {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f})")
            axis.fill_between(grid, np.maximum(0, average-spread), np.minimum(1, average+spread), color=colors[condition], alpha=.16)
        _style(axis, "CheXpert validation: mean ROC across 3 seeds")
    def plot_external(axis: plt.Axes) -> None:
        for condition, frame in _external_frames().items():
            fpr, tpr, _ = roc_curve(frame.y_true, frame.y_prob)
            auc = roc_auc_score(frame.y_true, frame.y_prob)
            axis.plot(fpr, tpr, color=colors[condition], linewidth=2, label=f"{condition.replace('_', ' ')} (AUROC {auc:.3f})")
        _style(axis, "RSNA external: frozen seed-42 models")
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    plot_internal(axis)
    _save(figure, output, "chexpert_internal_roc_comparison")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    plot_external(axis)
    _save(figure, output, "rsna_external_roc_comparison")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    plot_internal(axes[0])
    plot_external(axes[1])
    figure.tight_layout()
    _save(figure, output, "chexpert_rsna_roc_comparison")
    plt.close(figure)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-output", type=Path, default=ROOT / "configs" / "operating_thresholds")
    parser.add_argument("--analysis-output", type=Path, default=ROOT / "results" / "phase1_1_threshold_analysis")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of this script's derived artifacts only.")
    args = parser.parse_args()
    records, internal = derive_thresholds(args.threshold_output, args.overwrite)
    args.analysis_output.mkdir(parents=True, exist_ok=True)
    sensitivity_table(records, internal, args.analysis_output, args.overwrite)
    roc_figures(internal, args.analysis_output, args.overwrite)


if __name__ == "__main__":
    main()
