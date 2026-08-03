"""Publication-oriented, CPU-only analysis of completed controlled A4 runs.

This script reads small JSON/CSV result artifacts only.  It never opens a
checkpoint, image, or GPU device.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from scripts.classification.checkpoint_compatibility import (  # noqa: E402
    classify_checkpoint_configuration, normalize_checkpoint_configuration,
)

SEEDS = (42, 123, 2026)
PRIMARY_METRICS = ("auroc", "pr_auc", "fixed_accuracy", "fixed_balanced_accuracy",
                   "fixed_f1", "fixed_precision", "fixed_sensitivity", "fixed_specificity")
SECONDARY_METRICS = ("max_f1_threshold", "max_f1_f1", "max_f1_balanced_accuracy",
                     "max_f1_sensitivity", "max_f1_specificity")
REQUIRED_METRICS = ("auroc", "pr_auc", "fixed_threshold", *PRIMARY_METRICS[2:], *SECONDARY_METRICS)
ALL_METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS)
CONFIG_FIELDS = ("backbone", "preprocessing", "input_size", "pretrained", "augmentation",
                 "rotation_degrees", "horizontal_flip", "loss", "optimizer", "learning_rate",
                 "backbone_learning_rate", "head_learning_rate", "weight_decay", "scheduler",
                 "batch_size", "epochs", "early_stopping_patience")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_config(raw: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Normalize only recognised legacy A4 metadata; reject unknown schemas."""
    try:
        schema = classify_checkpoint_configuration(raw)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    config = normalize_checkpoint_configuration(raw) if schema == "legacy_optimisation" else dict(raw)
    if "input_size" not in config and "classifier_image_size" in config:
        config["input_size"] = config["classifier_image_size"]
    missing = [field for field in ("experiment", "input_mode", "seed", *CONFIG_FIELDS) if field not in config]
    if missing:
        raise ValueError(f"{source}: missing required configuration fields: {', '.join(missing)}")
    try:
        config["seed"] = int(config["seed"])
        config["input_size"] = int(config["input_size"])
        config["epochs"] = int(config["epochs"])
        config["batch_size"] = int(config["batch_size"])
        config["early_stopping_patience"] = int(config["early_stopping_patience"])
        config["rotation_degrees"] = float(config["rotation_degrees"])
        for field in ("learning_rate", "backbone_learning_rate", "head_learning_rate", "weight_decay"):
            config[field] = float(config[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid numeric configuration value: {error}") from error
    if config["input_mode"] not in {"original", "hard_masked"}:
        raise ValueError(f"{source}: input_mode must be original or hard_masked")
    if not isinstance(config["pretrained"], bool) or not isinstance(config["horizontal_flip"], bool):
        raise ValueError(f"{source}: pretrained and horizontal_flip must be booleans")
    if config["input_size"] < 1 or config["batch_size"] < 1 or config["epochs"] < 1 or config["early_stopping_patience"] < 0:
        raise ValueError(f"{source}: invalid image size, batch size, epoch budget, or patience")
    return config


def load_run(directory: str | Path, expected_model: str, expected_seed: int) -> dict[str, Any]:
    directory = Path(directory)
    errors: list[str] = []
    if not directory.is_dir():
        raise ValueError(f"{expected_model} seed {expected_seed}: run directory does not exist: {directory}")
    required = [directory / name for name in ("validation_metrics.json", "config.json", "run_summary.json")]
    absent = [str(path.name) for path in required if not path.is_file()]
    if absent:
        raise ValueError(f"{directory}: missing required files: {', '.join(absent)}")
    metrics, summary = _json(required[0]), _json(required[2])
    config = canonical_config(_json(required[1]), required[1])
    missing_metrics = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing_metrics:
        errors.append("missing required validation metrics: " + ", ".join(missing_metrics))
    for name in REQUIRED_METRICS:
        if name in metrics:
            try:
                if not math.isfinite(float(metrics[name])):
                    errors.append(f"metric {name} is not finite")
            except (TypeError, ValueError):
                errors.append(f"metric {name} is not numeric")
    if config["seed"] != expected_seed:
        errors.append(f"config seed={config['seed']!r}, expected {expected_seed}")
    expected_mode = "original" if expected_model == "original" else "hard_masked"
    if config["input_mode"] != expected_mode:
        errors.append(f"input_mode={config['input_mode']!r}, expected {expected_mode!r}")
    if errors:
        raise ValueError(f"{directory}: " + "; ".join(errors))
    return {"directory": directory, "model": expected_model, "seed": expected_seed,
            "config": config, "metrics": {key: float(metrics[key]) for key in REQUIRED_METRICS},
            "best_epoch": summary.get("best_epoch"), "experiment": str(config["experiment"]),
            "predictions": directory / "validation_predictions.csv"}


def validate_runs(runs: list[dict[str, Any]]) -> None:
    """Report all controlled-comparison violations in a single exception."""
    problems: list[str] = []
    for model in ("original", "hard_masked"):
        found = {run["seed"] for run in runs if run["model"] == model}
        if found != set(SEEDS):
            problems.append(f"{model} seeds are {sorted(found)}, expected {list(SEEDS)}")
    for seed in SEEDS:
        pair = {run["model"]: run for run in runs if run["seed"] == seed}
        if set(pair) != {"original", "hard_masked"}:
            problems.append(f"seed {seed}: missing original/masked pair")
            continue
        left, right = pair["original"]["config"], pair["hard_masked"]["config"]
        for field in CONFIG_FIELDS:
            if left[field] != right[field]:
                problems.append(f"seed {seed}: configuration mismatch {field}: original={left[field]!r}, hard_masked={right[field]!r}")
        if left["input_mode"] != "original" or right["input_mode"] != "hard_masked":
            problems.append(f"seed {seed}: input modes are not original/hard_masked")
        if left["experiment"] == right["experiment"]:
            problems.append(f"seed {seed}: original and hard_masked experiment names must differ")
    if problems:
        raise ValueError("Configuration comparability validation failed:\n- " + "\n- ".join(problems))


def summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        for metric in ALL_METRICS:
            values = group[metric].to_numpy(float)
            sd = float(np.std(values, ddof=1))
            rows.append({"model": model, "metric": metric, "n": len(values), "mean": float(np.mean(values)),
                         "sample_sd": sd, "standard_error": sd / math.sqrt(len(values)),
                         "minimum": float(np.min(values)), "maximum": float(np.max(values))})
    mean_sd = pd.DataFrame(rows)
    return mean_sd, mean_sd[["model", "metric", "n", "minimum", "maximum"]].copy()


def paired_statistics(differences: np.ndarray) -> dict[str, Any]:
    """Return exploratory paired inference, safely handling degenerate inputs.

    Constant paired differences have a meaningful descriptive mean and a
    degenerate CI, but standardised effects and t-tests are not evaluable.
    """
    raw_values = np.asarray(differences, dtype=float).reshape(-1)
    values = raw_values[np.isfinite(raw_values)]
    result: dict[str, Any] = {
        "n": int(len(values)), "n_non_finite_excluded": int(len(raw_values) - len(values)),
        "exploratory": True,
    }
    if len(values) < 2:
        result.update({"reason": "requires at least two finite paired differences"})
        return result
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    result.update({"mean_paired_difference": mean, "sample_sd_paired_difference": sd,
                   "minimum_difference": float(values.min()), "maximum_difference": float(values.max())})
    zero_variance = bool(np.isclose(sd, 0.0, rtol=1e-12, atol=1e-12))
    if zero_variance:
        result.update({"paired_t_p_value": None, "paired_t_reason": "zero variance in paired differences",
                       "cohens_dz": None, "hedges_gz": None,
                       "paired_t_status": "not_evaluable_zero_variance",
                       "effect_size_status": "not_evaluable_zero_variance",
                       "ci_lower": mean, "ci_upper": mean,
                       "ci_status": "degenerate_zero_variance"})
    else:
        se = sd / math.sqrt(len(values)); critical = float(stats.t.ppf(0.975, df=len(values)-1))
        test = stats.ttest_1samp(values, 0.0)
        dz = mean / sd
        result.update({"paired_t_statistic": float(test.statistic), "paired_t_p_value": float(test.pvalue),
                       "cohens_dz": dz, "hedges_gz": (1 - 3 / (4 * (len(values)-1) - 1)) * dz,
                       "ci_lower": mean - critical * se, "ci_upper": mean + critical * se})
    if np.all(values == 0):
        result.update({"wilcoxon_p_value": None, "wilcoxon_reason": "all paired differences are zero"})
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                test = stats.wilcoxon(values, alternative="two-sided", method="exact")
            result["wilcoxon_statistic"] = float(test.statistic)
            result["wilcoxon_p_value"] = float(test.pvalue)
        except Exception as error:  # scipy correctly refuses some zero/tie configurations
            result.update({"wilcoxon_p_value": None, "wilcoxon_reason": str(error)})
    return result


def range_error_bars(
    means: np.ndarray | list[float], lower_bounds: np.ndarray | list[float],
    upper_bounds: np.ndarray | list[float],
) -> np.ndarray:
    """Build Matplotlib-safe asymmetric range bars without round-off negatives."""
    means_array = np.asarray(means, dtype=np.float64).reshape(-1)
    lower_array = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper_array = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    if not (means_array.shape == lower_array.shape == upper_array.shape):
        raise ValueError("means and range bounds must have matching one-dimensional shapes")
    if not (np.isfinite(means_array).all() and np.isfinite(lower_array).all() and np.isfinite(upper_array).all()):
        raise ValueError("range error bars require finite means and bounds")
    lower_errors = np.maximum(0.0, means_array - lower_array)
    upper_errors = np.maximum(0.0, upper_array - means_array)
    yerr = np.vstack([lower_errors, upper_errors]).astype(float)
    if yerr.shape != (2, len(means_array)) or not np.isfinite(yerr).all() or (yerr < 0).any():
        raise ValueError("invalid non-negative asymmetric error bars")
    return yerr


def normalize_predictions(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    label = "label" if "label" in data else "binary_target" if "binary_target" in data else "original_label" if "original_label" in data else None
    required = {"patient_id", "study_id", "image_path", "probability"}
    if label is None or not required.issubset(data.columns):
        raise ValueError(f"{path}: unsupported prediction schema")
    out = data[["patient_id", "study_id", "image_path", label, "probability"]].copy()
    out.columns = ["patient_id", "study_id", "image_path", "label", "probability"]
    out[["patient_id", "study_id", "image_path"]] = out[["patient_id", "study_id", "image_path"]].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["probability"] = pd.to_numeric(out["probability"], errors="raise").astype(float)
    if out.duplicated(["patient_id", "study_id", "image_path"]).any() or not np.isfinite(out.probability).all():
        raise ValueError(f"{path}: duplicate cases or non-finite probabilities")
    return out


def prediction_check(original: Path, masked: Path, threshold: float) -> dict[str, Any]:
    if not original.is_file() or not masked.is_file():
        return {"available": False, "reason": "validation_predictions.csv absent for one or both paired runs"}
    try:
        left, right = normalize_predictions(original), normalize_predictions(masked)
        keys = ["patient_id", "study_id", "image_path"]
        merged = left.merge(right, on=keys, how="outer", suffixes=("_original", "_hard_masked"), indicator=True)
        if not merged["_merge"].eq("both").all():
            return {"available": False, "reason": "validation case sets differ"}
        if not merged.label_original.eq(merged.label_hard_masked).all():
            return {"available": False, "reason": "validation labels differ"}
        oc = merged.probability_original >= threshold; mc = merged.probability_hard_masked >= threshold
        correct_o = oc.eq(merged.label_original.astype(bool)); correct_m = mc.eq(merged.label_original.astype(bool))
        return {"available": True, "paired_case_count": int(len(merged)),
                "masking_helped_count": int((correct_m & ~correct_o).sum()), "masking_hurt_count": int((correct_o & ~correct_m).sum())}
    except Exception as error:
        return {"available": False, "reason": str(error)}


def _save(fig: plt.Figure, output: Path, name: str) -> list[str]:
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"; fig.savefig(path, dpi=300, bbox_inches="tight"); files.append(path.name)
    plt.close(fig); return files


def make_figures(frame: pd.DataFrame, paired: pd.DataFrame, tests: pd.DataFrame, output: Path) -> list[dict[str, Any]]:
    figures = []
    labels = {"auroc": "AUROC", "pr_auc": "PR AUC", "fixed_accuracy": "Fixed-threshold accuracy", "fixed_f1": "Fixed-threshold F1", "fixed_sensitivity": "Fixed-threshold sensitivity", "fixed_specificity": "Fixed-threshold specificity"}
    for metric, label in labels.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.2));
        for seed in SEEDS:
            row = paired[(paired.seed == seed) & (paired.metric == metric)].iloc[0]
            ax.plot([0, 1], [row.original, row.hard_masked], color="0.65", marker="o", label=f"Seed {seed}")
        ax.set_xticks([0, 1], ["Original", "Hard-masked"]); ax.set_ylim(0, 1); ax.set_ylabel(label)
        ax.set_title(f"{label} across matched seeds (higher is better)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
        name = f"{metric}_across_seeds"; figures.append({"name": name, "files": _save(fig, output, name), "description": "Matched seed-level comparison; higher is better."})
    forest = tests[tests.metric.isin(PRIMARY_METRICS)]
    fig, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(forest))
    means = forest.mean_paired_difference.to_numpy(float); lo = forest.ci_lower.to_numpy(float); hi = forest.ci_upper.to_numpy(float)
    ax.errorbar(means, y, xerr=[means-lo, hi-means], fmt="o", color="#1565c0", capsize=3); ax.axvline(0, color="black", lw=1)
    for index, metric in enumerate(forest.metric):
        points = paired[paired.metric == metric].difference_masked_minus_original.to_numpy(float)
        ax.scatter(points, np.full(len(points), index) + np.linspace(-.12, .12, len(points)), color="#90caf9", zorder=3, label="Seed-level difference" if index == 0 else None)
    ax.set_yticks(y, forest.metric); ax.set_xlabel("Hard-masked minus original (positive favors hard-masked)"); ax.set_title("Exploratory paired mean differences (95% t CI, df=2)"); ax.grid(axis="x", alpha=.25)
    figures.append({"name": "paired_metric_difference_forest", "files": _save(fig, output, "paired_metric_difference_forest"), "description": "Exploratory paired differences; positive favors hard-masked."})
    fig, ax = plt.subplots(figsize=(9, 5));
    for model, color in [("original", "#555555"), ("hard_masked", "#d95f02")]:
        group = frame[frame.model == model]
        means = np.asarray([group[m].mean() for m in PRIMARY_METRICS], dtype=np.float64)
        lower_bounds = np.asarray([group[m].min() for m in PRIMARY_METRICS], dtype=np.float64)
        upper_bounds = np.asarray([group[m].max() for m in PRIMARY_METRICS], dtype=np.float64)
        ax.errorbar(range(len(PRIMARY_METRICS)), means, yerr=range_error_bars(means, lower_bounds, upper_bounds), fmt="o-", label=model.replace("_", " "), color=color, capsize=3)
        for index, metric in enumerate(PRIMARY_METRICS):
            ax.scatter(np.full(len(group), index) + (-.08 if model == "original" else .08), group[metric], color=color, alpha=.65, s=24, zorder=3)
    ax.set_xticks(range(len(PRIMARY_METRICS)), PRIMARY_METRICS, rotation=35, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Metric value (range bars across seeds)"); ax.set_title("Multi-metric seed stability (higher is better)"); ax.legend(); ax.grid(axis="y", alpha=.25)
    figures.append({"name": "multimetric_seed_stability", "files": _save(fig, output, "multimetric_seed_stability"), "description": "Mean and seed range by model; higher is better."})
    return figures


def run_analysis(paths: Mapping[tuple[str, int], str | Path], output_directory: str | Path) -> Path:
    runs, errors = [], []
    for model in ("original", "hard_masked"):
        for seed in SEEDS:
            try: runs.append(load_run(paths[(model, seed)], model, seed))
            except (KeyError, ValueError) as error: errors.append(str(error))
    if errors: raise ValueError("Run validation failed:\n- " + "\n- ".join(errors))
    validate_runs(runs)
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    rows = [{"model": r["model"], "input_mode": r["config"]["input_mode"], "seed": r["seed"], "experiment": r["experiment"], "best_epoch": r["best_epoch"], **{m:r["metrics"][m] for m in ALL_METRICS}} for r in runs]
    frame = pd.DataFrame(rows); frame.to_csv(output / "all_runs_metrics.csv", index=False)
    mean_sd, ranges = summary_tables(frame); mean_sd.to_csv(output / "model_summary_mean_sd.csv", index=False); ranges.to_csv(output / "model_summary_range.csv", index=False)
    pair_rows=[]; test_rows=[]
    for metric in ALL_METRICS:
        original = frame[frame.model == "original"].set_index("seed").loc[list(SEEDS), metric]
        masked = frame[frame.model == "hard_masked"].set_index("seed").loc[list(SEEDS), metric]
        diff = masked.to_numpy(float) - original.to_numpy(float)
        pair_rows.extend({"seed":seed,"metric":metric,"original":float(o),"hard_masked":float(h),"difference_masked_minus_original":float(d)} for seed,o,h,d in zip(SEEDS,original,masked,diff))
        if metric in PRIMARY_METRICS: test_rows.append({"metric":metric, **paired_statistics(diff)})
    paired = pd.DataFrame(pair_rows); paired.to_csv(output / "paired_seed_differences.csv", index=False)
    tests = pd.DataFrame(test_rows); tests.to_csv(output / "paired_seed_tests.csv", index=False)
    paired_summary = (paired.groupby("metric", sort=False).difference_masked_minus_original
                      .agg(mean_paired_difference="mean", sample_sd_paired_difference="std",
                           minimum_difference="min", maximum_difference="max").reset_index())
    paired_summary.to_csv(output / "paired_seed_summary.csv", index=False)
    orig = mean_sd[mean_sd.model == "original"].set_index("metric"); hard = mean_sd[mean_sd.model == "hard_masked"].set_index("metric")
    publication = paired_summary.merge(tests, on=["metric", "mean_paired_difference", "sample_sd_paired_difference", "minimum_difference", "maximum_difference"], how="left"); publication["original_mean"]=[orig.loc[m,"mean"] for m in publication.metric]; publication["original_sd"]=[orig.loc[m,"sample_sd"] for m in publication.metric]; publication["hard_masked_mean"]=[hard.loc[m,"mean"] for m in publication.metric]; publication["hard_masked_sd"]=[hard.loc[m,"sample_sd"] for m in publication.metric]
    publication["original_mean_sd"] = publication.apply(lambda x:f"{x.original_mean:.3f} ± {x.original_sd:.3f}",axis=1); publication["hard_masked_mean_sd"] = publication.apply(lambda x:f"{x.hard_masked_mean:.3f} ± {x.hard_masked_sd:.3f}",axis=1)
    publication[["metric","original_mean","original_sd","hard_masked_mean","hard_masked_sd","mean_paired_difference","ci_lower","ci_upper","paired_t_p_value","wilcoxon_p_value","cohens_dz","original_mean_sd","hard_masked_mean_sd"]].rename(columns={"ci_lower":"paired_difference_ci_lower","ci_upper":"paired_difference_ci_upper"}).to_csv(output / "publication_table_multiseed.csv", index=False)
    checks = {}
    for seed in SEEDS:
        original_run = next(r for r in runs if r["model"] == "original" and r["seed"] == seed)
        masked_run = next(r for r in runs if r["model"] == "hard_masked" and r["seed"] == seed)
        original_threshold, masked_threshold = original_run["metrics"]["fixed_threshold"], masked_run["metrics"]["fixed_threshold"]
        checks[str(seed)] = ({"available": False, "reason": "saved fixed thresholds differ between paired runs"}
                             if original_threshold != masked_threshold else prediction_check(original_run["predictions"], masked_run["predictions"], original_threshold))
    figures=make_figures(frame, paired, tests, output)
    test_records = tests.astype(object).where(pd.notna(tests), None).to_dict(orient="records")
    report={"warning":"Seed-level inferential tests are exploratory because only three paired seeds are available.","seeds":list(SEEDS),"primary_metrics":list(PRIMARY_METRICS),"secondary_descriptive_metrics":list(SECONDARY_METRICS),"paired_tests":test_records,"prediction_level_checks":checks,"figures":figures}
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metadata={"analysis":"completed-results-only multiseed A4 comparison","no_gpu":True,"checkpoints_loaded":False,"images_loaded":False,"archive_contents":"CSV, JSON, PNG, and PDF only"}
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = output.parent / f"{output.name}_results.zip"
    allowed = {".csv", ".json", ".png", ".pdf"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed:
                bundle.write(path, path.relative_to(output))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    for model in ("original", "masked"):
        for seed in SEEDS: parser.add_argument(f"--{model}-seed-{seed}", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args=build_parser().parse_args(argv)
    paths={("original",s):getattr(args,f"original_seed_{s}") for s in SEEDS}
    paths.update({("hard_masked",s):getattr(args,f"masked_seed_{s}") for s in SEEDS})
    try: run_analysis(paths,args.output_directory)
    except (ValueError, OSError) as error: raise SystemExit(f"ERROR: {error}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
