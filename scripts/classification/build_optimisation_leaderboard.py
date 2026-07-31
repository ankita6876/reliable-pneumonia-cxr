"""Build a validation-only leaderboard from completed optimisation runs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def build_leaderboard(output_root: Path) -> pd.DataFrame:
    """Read completed run summaries only; no manifests, images, or test outputs are read."""
    rows = []
    for directory in sorted(path for path in output_root.iterdir() if path.is_dir()):
        summary_path, config_path = (
            directory / "run_summary.json",
            directory / "config.json",
        )
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "experiment": config.get("experiment", directory.name),
                "status": summary.get("status"),
                "seed": config.get("seed"), "backbone": config.get("backbone"),
                "pretrained": config.get("pretrained"), "augmentation": config.get("augmentation"),
                "horizontal_flip": config.get("horizontal_flip", config.get("augmentation") == "historical"),
                "rotation_degrees": config.get("rotation_degrees", 7), "loss": config.get("loss"),
                "optimizer": config.get("optimizer", config.get("optimiser")),
                "learning_rate": config.get("learning_rate", config.get("head_learning_rate")),
                "weight_decay": config.get("weight_decay"), "epochs_requested": config.get("epochs"),
                "epochs_completed": summary.get("epochs_completed"),
                "best_epoch": summary.get("best_epoch"),
                "validation_auroc": summary.get("auroc"),
                "validation_pr_auc": summary.get("pr_auc"),
                "validation_f1_at_0_5": summary.get("fixed_f1", _f1_at_half(directory)),
                "validation_sensitivity_at_0_5": summary.get("fixed_sensitivity"),
                "validation_specificity_at_0_5": summary.get("fixed_specificity"),
                "validation_balanced_accuracy_at_0_5": summary.get("fixed_balanced_accuracy"),
                "max_f1_threshold": summary.get("max_f1_threshold", summary.get("selected_threshold")),
                "validation_f1_at_max_f1_threshold": summary.get("max_f1_f1", summary.get("selected_f1", summary.get("f1"))),
                "validation_sensitivity_at_max_f1_threshold": summary.get("max_f1_sensitivity", summary.get("selected_sensitivity", summary.get("sensitivity"))),
                "validation_specificity_at_max_f1_threshold": summary.get("max_f1_specificity", summary.get("selected_specificity", summary.get("specificity"))),
                "validation_balanced_accuracy_at_max_f1_threshold": summary.get("max_f1_balanced_accuracy", summary.get("selected_balanced_accuracy", summary.get("balanced_accuracy"))),
                "training_time_seconds": summary.get("training_time_seconds"),
                "checkpoint_path": str(directory / "best_checkpoint.pt"),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["validation_auroc", "validation_pr_auc", "validation_balanced_accuracy_at_0_5"],
            ascending=False,
            kind="stable",
        )
    table.to_csv(output_root / "experiment_leaderboard.csv", index=False)
    return table


def _f1_at_half(directory: Path) -> float | None:
    predictions = (
        pd.read_csv(directory / "validation_predictions.csv")
        if (directory / "validation_predictions.csv").is_file()
        else None
    )
    if predictions is None:
        return None
    y, p = predictions.label.to_numpy(), predictions.probability.to_numpy()
    tp = ((p >= 0.5) & (y == 1)).sum()
    fp = ((p >= 0.5) & (y == 0)).sum()
    fn = ((p < 0.5) & (y == 1)).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "outputs"
        / "classification_optimisation",
    )
    args = parser.parse_args()
    print(build_leaderboard(args.output_root).to_string(index=False))
