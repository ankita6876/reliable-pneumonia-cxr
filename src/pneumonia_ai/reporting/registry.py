"""Compact reporting that never loads checkpoints or prediction CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def build_registry(inputs: list[Path | str]) -> pd.DataFrame:
    """Discover manifests recursively and retain every valid or malformed run."""
    rows: list[dict[str, Any]] = []
    for root in sorted(map(Path, inputs)):
        manifests = sorted(root.rglob("run_manifest.json"))
        if not manifests:
            rows.append(
                {
                    "run_id": None,
                    "status": "invalid",
                    "missing_artifact_warnings": "missing manifest",
                }
            )
        for manifest_path in manifests:
            row = _read_run(manifest_path.parent, manifest_path)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "run_id" in frame:
        duplicates = frame["run_id"].notna() & frame["run_id"].duplicated(keep=False)
        frame.loc[duplicates, "status"] = "invalid"
        warnings = frame.loc[duplicates, "missing_artifact_warnings"].fillna("")
        frame.loc[duplicates, "missing_artifact_warnings"] = (
            warnings + " duplicate run ID"
        )
    return frame.sort_values(["run_id"], na_position="last").reset_index(drop=True)


def _read_run(directory: Path, manifest_path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "run_id": directory.name,
            "status": "invalid",
            "missing_artifact_warnings": f"malformed manifest: {error}",
        }
    config = _read_yaml(directory / "config.yaml", warnings)
    history = _read_history(directory / "training_history.csv", warnings)
    metrics = _read_json(directory / "metrics.json", warnings)
    checkpoint = directory / "best_validation_auroc.pt"
    if not checkpoint.exists():
        warnings.append("missing checkpoint")
    training = config.get("training", {}) if isinstance(config, dict) else {}
    model = config.get("model", {}) if isinstance(config, dict) else {}
    best: Any = {}
    if not history.empty and "validation_auroc" in history:
        best = history.loc[history["validation_auroc"].idxmax()]
    evaluation = metrics.get("validation", metrics) if isinstance(metrics, dict) else {}
    if not all(np.isfinite(value) for value in _numeric_values(evaluation)):
        warnings.append("non-finite evaluation metric")
    return _row(
        directory,
        manifest,
        model,
        training,
        best,
        evaluation,
        checkpoint,
        warnings,
    )


def _row(
    directory: Path,
    manifest: dict[str, Any],
    model: dict[str, Any],
    training: dict[str, Any],
    best: Any,
    evaluation: dict[str, Any],
    checkpoint: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Create the portable registry row from already-read metadata."""
    return {
        "run_id": manifest.get("run_id"),
        "timestamp": manifest.get("utc_timestamp"),
        "git_commit": manifest.get("git_commit_hash"),
        "dirty_tree": manifest.get("git_dirty_working_tree"),
        "model": manifest.get("model_name", model.get("name")),
        "preprocessing": model.get("preprocessing"),
        "label_strategy": manifest.get("label_strategy"),
        "seed": manifest.get("random_seed"),
        "dataset": manifest.get("dataset", "CheXpert"),
        "train_sample_count": manifest.get("train_sample_count"),
        "validation_sample_count": manifest.get("validation_sample_count"),
        "best_epoch": best.get("epoch") if hasattr(best, "get") else None,
        "best_validation_auroc": (
            best.get("validation_auroc") if hasattr(best, "get") else None
        ),
        "best_validation_auprc": (
            best.get("validation_auprc") if hasattr(best, "get") else None
        ),
        "validation_loss": (
            best.get("validation_loss") if hasattr(best, "get") else None
        ),
        "learning_rate": (
            best.get("learning_rate", training.get("learning_rate"))
            if hasattr(best, "get")
            else training.get("learning_rate")
        ),
        "class_weighting": training.get("class_weighting"),
        "amp": manifest.get("amp_enabled"),
        "checkpoint_size": checkpoint.stat().st_size if checkpoint.exists() else None,
        "run_directory_size": sum(
            path.stat().st_size
            for path in directory.rglob("*")
            if path.is_file()
        ),
        "evaluation_auroc": evaluation.get("auroc"),
        "evaluation_auprc": evaluation.get("auprc"),
        "ece": evaluation.get("expected_calibration_error"),
        "brier_score": evaluation.get("brier_score"),
        "nll": evaluation.get("negative_log_likelihood"),
        "error_detection_auroc": evaluation.get("error_detection_auroc"),
        "area_under_risk_coverage": _risk_area(directory),
        "status": "valid" if not warnings else "warning",
        "missing_artifact_warnings": chr(59).join(warnings),
    }


def compare_registry(registry: pd.DataFrame, output: Path | str) -> None:
    """Write validation-only ranked and grouped comparison CSVs."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    valid = registry.loc[registry.status != "invalid"].copy()
    ranked = valid.sort_values(
        ["best_validation_auprc", "best_validation_auroc"],
        ascending=False,
    )
    ranked.to_csv(destination / "ranked_model_comparison.csv", index=False)
    tables = {
        "label_strategy_comparison.csv": ["label_strategy"],
        "seed_variation_summary.csv": ["model", "label_strategy"],
        "calibration_comparison.csv": [
            "model",
            "label_strategy",
            "ece",
            "brier_score",
            "nll",
        ],
        "selective_prediction_comparison.csv": [
            "model",
            "label_strategy",
            "area_under_risk_coverage",
        ],
    }
    for name, columns in tables.items():
        groups = columns[: min(2, len(columns))]
        table = valid.loc[:, columns]
        grouped = table.groupby(groups, dropna=False)
        summary = grouped.mean(numeric_only=True).reset_index()
        summary.to_csv(destination / name, index=False)


def generate_publication_artifacts(registry: pd.DataFrame, output: Path | str) -> None:
    """Create compact publication tables and headless validation-only figures."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    valid = registry.loc[registry.status != "invalid"].copy()
    tables = {
        "dataset_summary.csv": ["dataset"],
        "internal_validation_models.csv": [
            "model",
            "best_validation_auroc",
            "best_validation_auprc",
        ],
        "uncertain_label_ablation.csv": [
            "label_strategy",
            "best_validation_auroc",
            "best_validation_auprc",
        ],
        "calibration_table.csv": ["model", "ece", "brier_score", "nll"],
        "uncertainty_failure_table.csv": ["model", "error_detection_auroc"],
        "selective_prediction_table.csv": ["model", "area_under_risk_coverage"],
    }
    for filename, columns in tables.items():
        table = valid.loc[:, columns]
        table.to_csv(destination / filename, index=False)
    _bar(
        valid,
        "model",
        "best_validation_auroc",
        destination / "validation_auroc_comparison.png",
    )
    _bar(
        valid,
        "model",
        "best_validation_auprc",
        destination / "validation_auprc_comparison.png",
    )
    _bar(valid, "model", "ece", destination / "calibration_comparison.png")
    _bar(
        valid,
        "model",
        "area_under_risk_coverage",
        destination / "risk_coverage_comparison.png",
    )


def _bar(frame: pd.DataFrame, category: str, value: str, output: Path) -> None:
    plot = frame.dropna(subset=[value])
    figure, axis = plt.subplots()
    try:
        axis.bar(plot[category].astype(str), plot[value])
        axis.set_xlabel(category)
        axis.set_ylabel(value)
        figure.tight_layout()
        figure.savefig(output, dpi=300)
    finally:
        plt.close(figure)


def _read_yaml(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append("missing config")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        warnings.append("malformed config")
        return {}


def _read_history(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append("missing history")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        warnings.append("malformed history")
        return pd.DataFrame()


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append("missing metrics")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.append("malformed metrics")
        return {}


def _numeric_values(data: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for value in data.values()
        if isinstance(value, (float, int))
    ]


def _risk_area(directory: Path) -> float | None:
    path = directory / "risk_coverage.csv"
    if not path.exists():
        return None
    curve = pd.read_csv(path)
    if {"risk", "coverage"} <= set(curve):
        return float(np.trapz(curve["risk"], curve["coverage"]))
    return None
