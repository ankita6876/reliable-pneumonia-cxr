from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"C:\Research\outputs\classification_ablation")
MODES = ["original", "hard_masked", "lung_crop"]
N_BOOTSTRAPS = 5000
SEED = 42


def load_predictions(mode: str) -> pd.DataFrame:
    path = ROOT / mode / "predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = ["image_path", "binary_target", "probability"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(
            f"{path} is missing columns: {', '.join(missing)}"
        )

    result = df[required].copy()
    result = result.rename(
        columns={
            "image_path": "sample_id",
            "binary_target": "label",
            "probability": f"probability_{mode}",
        }
    )

    result["sample_id"] = result["sample_id"].astype(str)
    result["label"] = pd.to_numeric(
        result["label"], errors="raise"
    ).astype(int)

    result[f"probability_{mode}"] = pd.to_numeric(
        result[f"probability_{mode}"],
        errors="raise",
    ).astype(float)

    if result["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate image paths found in {mode}")

    probabilities = result[f"probability_{mode}"]

    if probabilities.isna().any():
        raise ValueError(f"Missing probabilities found in {mode}")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"Probabilities outside [0,1] found in {mode}"
        )

    print(
        f"{mode}: {len(result)} predictions loaded"
    )

    return result


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def bootstrap_p_value(differences: np.ndarray) -> float:
    below_or_equal_zero = np.mean(differences <= 0)
    above_or_equal_zero = np.mean(differences >= 0)

    return float(
        min(
            1.0,
            2.0 * min(
                below_or_equal_zero,
                above_or_equal_zero,
            ),
        )
    )


frames = {
    mode: load_predictions(mode)
    for mode in MODES
}

merged = frames["original"]

for mode in MODES[1:]:
    merged = merged.merge(
        frames[mode],
        on=["sample_id", "label"],
        how="inner",
        validate="one_to_one",
    )

expected_rows = len(frames["original"])

if len(merged) != expected_rows:
    raise ValueError(
        f"Only {len(merged)} of {expected_rows} rows aligned."
    )

y_true = merged["label"].to_numpy()

if set(np.unique(y_true)) != {0, 1}:
    raise ValueError(
        f"Expected binary labels 0 and 1. Found: {np.unique(y_true)}"
    )

print("\nAligned samples:", len(merged))
print("Positive samples:", int(y_true.sum()))
print("Negative samples:", int((y_true == 0).sum()))

point_estimates = {}

for mode in MODES:
    probabilities = merged[f"probability_{mode}"].to_numpy()

    point_estimates[mode] = {
        "auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }

rng = np.random.default_rng(SEED)
sample_count = len(merged)

bootstrap_values = {
    mode: {
        "auc": [],
        "pr_auc": [],
    }
    for mode in MODES
}

completed = 0

while completed < N_BOOTSTRAPS:
    indices = rng.integers(
        0,
        sample_count,
        size=sample_count,
    )

    sampled_labels = y_true[indices]

    if len(np.unique(sampled_labels)) < 2:
        continue

    for mode in MODES:
        probabilities = merged[
            f"probability_{mode}"
        ].to_numpy()[indices]

        bootstrap_values[mode]["auc"].append(
            roc_auc_score(
                sampled_labels,
                probabilities,
            )
        )

        bootstrap_values[mode]["pr_auc"].append(
            average_precision_score(
                sampled_labels,
                probabilities,
            )
        )

    completed += 1

summary_rows = []

for mode in MODES:
    auc_values = np.asarray(
        bootstrap_values[mode]["auc"]
    )
    pr_values = np.asarray(
        bootstrap_values[mode]["pr_auc"]
    )

    auc_low, auc_high = confidence_interval(auc_values)
    pr_low, pr_high = confidence_interval(pr_values)

    summary_rows.append(
        {
            "mode": mode,
            "n": sample_count,
            "auc": point_estimates[mode]["auc"],
            "auc_ci_lower": auc_low,
            "auc_ci_upper": auc_high,
            "pr_auc": point_estimates[mode]["pr_auc"],
            "pr_auc_ci_lower": pr_low,
            "pr_auc_ci_upper": pr_high,
        }
    )

pairs = [
    ("hard_masked", "original"),
    ("lung_crop", "original"),
    ("hard_masked", "lung_crop"),
]

comparison_rows = []

for model_1, model_2 in pairs:
    for metric in ["auc", "pr_auc"]:
        values_1 = np.asarray(
            bootstrap_values[model_1][metric]
        )
        values_2 = np.asarray(
            bootstrap_values[model_2][metric]
        )

        differences = values_1 - values_2
        lower, upper = confidence_interval(differences)

        observed_difference = (
            point_estimates[model_1][metric]
            - point_estimates[model_2][metric]
        )

        comparison_rows.append(
            {
                "metric": metric,
                "model_1": model_1,
                "model_2": model_2,
                "model_1_value":
                    point_estimates[model_1][metric],
                "model_2_value":
                    point_estimates[model_2][metric],
                "difference":
                    observed_difference,
                "difference_ci_lower": lower,
                "difference_ci_upper": upper,
                "p_value":
                    bootstrap_p_value(differences),
            }
        )

summary_df = pd.DataFrame(summary_rows)
comparison_df = pd.DataFrame(comparison_rows)

summary_path = ROOT / "bootstrap_metric_summary.csv"
comparison_path = ROOT / "paired_bootstrap_comparison.csv"
aligned_path = ROOT / "aligned_test_predictions.csv"

summary_df.to_csv(summary_path, index=False)
comparison_df.to_csv(comparison_path, index=False)
merged.to_csv(aligned_path, index=False)

print("\n=== BOOTSTRAP SUMMARY ===")
print(summary_df.round(4).to_string(index=False))

print("\n=== PAIRED COMPARISONS ===")
print(comparison_df.round(4).to_string(index=False))

print("\nCreated:")
print(summary_path)
print(comparison_path)
print(aligned_path)
