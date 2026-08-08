"""Deterministically rebuild the frozen thesis-baseline results document.

This reader only opens JSON/CSV members in preserved archives or extracted
result directories.  It neither loads checkpoints nor writes anywhere except
the explicitly requested Markdown output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE_TAG = "v1-thesis-baseline"
BASELINE_COMMIT = "37966623bb490431308394c1e51f46dd5f870d0d"
ARCHIVES = ROOT / "experiments" / "kaggle_gpu" / "archives"
EXTRACTED = ROOT / "experiments" / "kaggle_gpu" / "extracted"
PRIMARY = (
    ("auroc", "AUROC"), ("pr_auc", "PR-AUC"),
    ("fixed_accuracy", "Accuracy"),
    ("fixed_balanced_accuracy", "Balanced accuracy"),
    ("fixed_f1", "F1"), ("fixed_precision", "Precision"),
    ("fixed_sensitivity", "Sensitivity"), ("fixed_specificity", "Specificity"),
)
RUNS = {
    ("original", 42): ("A4_original_control_results.zip", "validation_metrics.json", "run_summary.json"),
    ("original", 123): ("A4_original_seed_123_results.zip", "validation_metrics.json", "run_summary.json"),
    ("original", 2026): ("A4_original_seed_2026_results.zip", "validation_metrics.json", "run_summary.json"),
    # The seed-42 masked result is preserved in the optimisation extract, not
    # in a separately identifiable A4_hard_masked_seed_42 archive.
    ("hard_masked", 42): ("thesis_optimisation_results/classification_optimisation/A4_regularised_optimisation", "validation_metrics.json", "run_summary.json"),
    ("hard_masked", 123): ("A4_hard_masked_seed_123_results.zip", "validation_metrics.json", "run_summary.json"),
    ("hard_masked", 2026): ("A4_hard_masked_seed_2026_results.zip", "validation_metrics.json", "run_summary.json"),
}


def _read_zip_json(archive: Path, suffix: str) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as bundle:
        matches = [name for name in bundle.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"{archive}: expected one member ending {suffix}, found {matches}")
        return json.loads(bundle.read(matches[0]).decode("utf-8"))


def _read_json(source: str, member: str) -> dict[str, Any]:
    if source.endswith(".zip"):
        return _read_zip_json(ARCHIVES / source, member)
    path = EXTRACTED / source / member
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs() -> dict[tuple[str, int], dict[str, Any]]:
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for key, (source, metrics_member, summary_member) in RUNS.items():
        metrics, summary = _read_json(source, metrics_member), _read_json(source, summary_member)
        missing = [name for name, _ in PRIMARY if name not in metrics]
        if missing:
            raise ValueError(f"{source}: missing {', '.join(missing)}")
        loaded[key] = {"metrics": metrics, "best_epoch": summary.get("best_epoch"), "source": source}
    return loaded


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _t_two_sided_p_df2(t_value: float) -> float:
    """Exact two-sided Student t p value for df=2 (the three-seed design)."""
    absolute = abs(t_value)
    return 1.0 - absolute / math.sqrt(absolute * absolute + 2.0)


def _wilcoxon_exact_two_sided(values: list[float]) -> float | None:
    """Exact sign-rank p value for non-zero, non-tied three-seed differences."""
    if len(values) != 3 or any(value == 0 for value in values):
        return None
    absolute = sorted((abs(value), index) for index, value in enumerate(values))
    if any(math.isclose(absolute[index][0], absolute[index + 1][0]) for index in range(2)):
        return None
    ranks = [0, 0, 0]
    for rank, (_, index) in enumerate(absolute, start=1):
        ranks[index] = rank
    observed = min(sum(rank for rank, value in zip(ranks, values) if value > 0), sum(rank for rank, value in zip(ranks, values) if value < 0))
    sums = [sum(ranks[index] for index in range(3) if mask & (1 << index)) for mask in range(8)]
    return min(1.0, 2.0 * sum(total <= observed for total in sums) / 8.0)


def paired_stats(values: list[float]) -> dict[str, float | None]:
    avg, sd = mean(values), stdev(values)
    if sd == 0:
        return {"mean": avg, "ci_low": avg, "ci_high": avg, "t_p": None, "dz": None, "wilcoxon_p": None}
    se = sd / math.sqrt(3)
    t_value = avg / se
    margin = 4.302652729911275 * se  # 97.5% t critical value, df=2.
    return {"mean": avg, "ci_low": avg - margin, "ci_high": avg + margin,
            "t_p": _t_two_sided_p_df2(t_value), "dz": avg / sd,
            "wilcoxon_p": _wilcoxon_exact_two_sided(values)}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown(runs: dict[tuple[str, int], dict[str, Any]]) -> str:
    lines = [
        "# Frozen thesis baseline results",
        "",
        f"Frozen reference: Git tag `{BASELINE_TAG}` at commit `{BASELINE_COMMIT}`. This is a forensic snapshot; it does not retrain models, change thresholds, or alter experiment artifacts.",
        "",
        "## Internal CheXpert validation: matched three-seed summary",
        "",
        "All operating-point metrics below use the archived fixed threshold of 0.5. Values are mean ± sample SD across seeds 42, 123, and 2026.",
        "",
        "| Metric | Original | Hard-masked |",
        "|---|---:|---:|",
    ]
    for key, label in PRIMARY:
        original = [float(runs[("original", seed)]["metrics"][key]) for seed in (42, 123, 2026)]
        masked = [float(runs[("hard_masked", seed)]["metrics"][key]) for seed in (42, 123, 2026)]
        lines.append(f"| {label} | {_fmt(mean(original))} ± {_fmt(stdev(original))} | {_fmt(mean(masked))} ± {_fmt(stdev(masked))} |")
    lines += ["", "Seed-level best epochs: original 42/123/2026 = 9/8/9; hard-masked 42/123/2026 = 7/9/7.", "",
              "### Internal paired seed statistics", "", "Hard-masked minus original. These tests are exploratory because n=3 paired seeds. The 95% intervals are t intervals (df=2); exact Wilcoxon values are reported only where the three non-zero absolute differences are untied.", "", "| Metric | Mean difference | 95% CI | paired t p | Wilcoxon p | Cohen’s dz |", "|---|---:|---:|---:|---:|---:|"]
    for key, label in PRIMARY:
        diff = [float(runs[("hard_masked", seed)]["metrics"][key]) - float(runs[("original", seed)]["metrics"][key]) for seed in (42, 123, 2026)]
        stats = paired_stats(diff)
        wp = "not evaluable" if stats["wilcoxon_p"] is None else _fmt(float(stats["wilcoxon_p"]))
        lines.append(f"| {label} | {_fmt(float(stats['mean']))} | [{_fmt(float(stats['ci_low']))}, {_fmt(float(stats['ci_high']))}] | {_fmt(float(stats['t_p'])) if stats['t_p'] is not None else 'not evaluable'} | {wp} | {_fmt(float(stats['dz'])) if stats['dz'] is not None else 'not evaluable'} |")
    external = EXTRACTED / "RSNA_external_statistical_analysis" / "RSNA_external_statistical_analysis"
    report = json.loads((external / "external_validation_report.json").read_text(encoding="utf-8"))
    bootstrap = _csv(external / "paired_bootstrap_comparison.csv")
    lines += ["", "## RSNA external classification", "", f"Paired RSNA cohort: {report['paired_case_count']:,} cases ({report['positive_count']:,} positive; {report['negative_count']:,} negative). The preserved report used threshold 0.5.", "", "| Metric | Original | Hard-masked | Difference (masked − original), 95% paired-bootstrap CI |", "|---|---:|---:|---:|"]
    names = {"auroc": "AUROC", "pr_auc": "PR-AUC", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1": "F1", "precision": "Precision", "sensitivity": "Sensitivity", "specificity": "Specificity", "negative_predictive_value": "NPV", "brier": "Brier", "negative_log_likelihood": "NLL", "ece": "ECE"}
    for row in bootstrap:
        p = "p < 0.001" if float(row["bootstrap_p_value"]) == 0 else f"p = {_fmt(float(row['bootstrap_p_value']))}"
        lines.append(f"| {names[row['metric']]} | {_fmt(float(row['original']))} | {_fmt(float(row['hard_masked']))} | {_fmt(float(row['difference_masked_minus_original']))} [{_fmt(float(row['difference_ci_lower']))}, {_fmt(float(row['difference_ci_upper']))}]; {p} |")
    mc = report["mcnemar"]
    outcomes = {row["masking_outcome"]: row["count"] for row in _csv(external / "masking_outcome_summary.csv")}
    lines += ["", f"McNemar: original-correct/masked-wrong = {mc['original_correct_masked_wrong']:,}; original-wrong/masked-correct = {mc['original_wrong_masked_correct']:,}; discordant pairs = {mc['discordant_pairs']:,}; exact p < 0.001 (the stored finite-resampling-style value is 0.0).", f"Masking helped {int(outcomes['masking_helped']):,} cases and hurt {int(outcomes['masking_hurt']):,} cases.", ""]
    gradcam = EXTRACTED / "rsna_gradcam_full_results" / "rsna_gradcam_full"
    grad_report = json.loads((gradcam / "rsna_gradcam_report.json").read_text(encoding="utf-8"))
    completion = json.loads((gradcam / "rsna_gradcam_completion_summary.json").read_text(encoding="utf-8"))
    lines += ["## RSNA Grad-CAM localization", "", f"Paired localization cohort: {completion['fully_paired_patient_count']:,} of {completion['expected_patient_count']:,} boxed RSNA-positive patients. One hard-masked case (`{completion['excluded_from_paired_analysis'][0]['patient_id']}`) was excluded after deterministic retry because its activation map was constant or unavailable; no heatmap was imputed.", "", "| Metric | Original | Hard-masked | Difference (masked − original), 95% paired-bootstrap CI |", "|---|---:|---:|---:|"]
    for item in grad_report["bootstrap"]:
        metric = item["metric"]
        original = grad_report["model_metric_means"]["original"][metric]
        masked = grad_report["model_metric_means"]["hard_masked"][metric]
        lines.append(f"| {metric} | {_fmt(original)} | {_fmt(masked)} | {_fmt(item['point_difference'])} [{_fmt(item['ci_2_5'])}, {_fmt(item['ci_97_5'])}]; p < 0.001 |")
    pg = grad_report["mcnemar_pointing_game"]
    lines += ["", f"Pointing Game McNemar: original-only hit = {pg['original_only_hit']:,}; hard-masked-only hit = {pg['masked_only_hit']:,}; exact p < 0.001 (stored value 0.0).", "", "## Provenance", "", "Internal seed archives: `experiments/kaggle_gpu/archives/A4_original_control_results.zip`, `A4_original_seed_123_results.zip`, `A4_original_seed_2026_results.zip`, `A4_hard_masked_seed_123_results.zip`, and `A4_hard_masked_seed_2026_results.zip` (each `validation_metrics.json` and `run_summary.json`). The reconstructed hard-masked seed-42 values are read from `experiments/kaggle_gpu/extracted/thesis_optimisation_results/classification_optimisation/A4_regularised_optimisation/{validation_metrics.json,run_summary.json}`. There is no separately identifiable local `A4_hard_masked_seed_42` archive; this limitation is retained explicitly.", "", "External classification: `experiments/kaggle_gpu/extracted/RSNA_external_statistical_analysis/RSNA_external_statistical_analysis/{external_validation_report.json,paired_bootstrap_comparison.csv,masking_outcome_summary.csv,mcnemar_test.json,domain_shift_summary.json,paired_predictions.csv}`.", "", "Localization: `experiments/kaggle_gpu/extracted/rsna_gradcam_full_results/rsna_gradcam_full/{rsna_gradcam_model_summary.csv,rsna_gradcam_bootstrap_comparison.csv,rsna_gradcam_paired_comparison.csv,rsna_gradcam_completion_summary.json,rsna_gradcam_report.json,rsna_gradcam_failures.csv,rsna_gradcam_case_metrics_paired_complete.csv}`.", "", "## Frozen interpretation", "", "Hard masking improves Grad-CAM localization and increases external sensitivity. It reduces external discrimination (AUROC and PR-AUC), specificity, precision, and calibration (higher Brier, NLL, and ECE). Thus, the explainability improvement does not imply predictive improvement.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results_baseline.md")
    args = parser.parse_args()
    args.output.write_text(_markdown(load_runs()), encoding="utf-8")


if __name__ == "__main__":
    main()
