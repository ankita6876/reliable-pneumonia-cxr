"""Generate publication-quality figures from aligned classifier predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .plotting import (
        configure_style,
        plot_calibration,
        plot_confusion_matrices,
        plot_precision_recall,
        plot_probability_distributions,
        plot_reliability_diagram,
        plot_roc,
        save_figure,
        validate_prediction_set,
    )
except ImportError:  # Direct script execution has no package context.
    from plotting import (
        configure_style,
        plot_calibration,
        plot_confusion_matrices,
        plot_precision_recall,
        plot_probability_distributions,
        plot_reliability_diagram,
        plot_roc,
        save_figure,
        validate_prediction_set,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "outputs" / "classification_ablation"
DEFAULT_LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification_ablation"
LABEL_CANDIDATES = ("label", "binary_target", "target", "y_true", "ground_truth")


def default_input_path() -> Path:
    """Locate the standard ablation output outside or inside the repository."""
    for root in (DEFAULT_OUTPUT_ROOT, DEFAULT_LOCAL_OUTPUT_ROOT):
        candidate = root / "aligned_test_predictions.csv"
        if candidate.is_file():
            return candidate
    return DEFAULT_OUTPUT_ROOT / "aligned_test_predictions.csv"


def parse_args() -> argparse.Namespace:
    """Parse figure-generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=default_input_path(), help="Aligned predictions CSV.")
    parser.add_argument("--output-directory", type=Path, help="Directory for PNG and PDF figures (default: sibling figures directory).")
    parser.add_argument("--label-column", help="Optional binary label column override.")
    parser.add_argument("--probability-columns", nargs="+", help="Optional prediction-column overrides.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for confusion matrices.")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of uniform calibration bins.")
    return parser.parse_args()


def detect_label_column(frame: pd.DataFrame, override: str | None = None) -> str:
    """Detect a binary target column, optionally using an explicit override."""
    if override is not None:
        if override not in frame.columns:
            raise KeyError(f"Requested label column is absent: {override}")
        return override
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    binary_columns = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(numeric) == len(frame) and set(numeric.unique()).issubset({0, 1}) and len(numeric.unique()) == 2:
            binary_columns.append(column)
    if len(binary_columns) == 1:
        return binary_columns[0]
    raise ValueError("Could not uniquely detect a binary label column; use --label-column.")


def detect_probability_columns(frame: pd.DataFrame, label_column: str, override: Sequence[str] | None = None) -> list[str]:
    """Detect model probability columns without requiring fixed model names."""
    if override is not None:
        missing = set(override).difference(frame.columns)
        if missing:
            raise KeyError(f"Requested probability columns are absent: {', '.join(sorted(missing))}")
        columns = list(override)
    else:
        named = [column for column in frame.columns if column.lower().startswith(("probability", "prob_", "prediction", "score"))]
        columns = named or [
            column for column in frame.columns
            if column != label_column
            and pd.api.types.is_numeric_dtype(frame[column])
            and frame[column].dropna().between(0, 1).all()
            and not set(frame[column].dropna().unique()).issubset({0, 1})
        ]
    if not columns:
        raise ValueError("No probability columns detected; use --probability-columns.")
    return columns


def generate_figures(input_csv: Path, output_directory: Path | None = None, label_column: str | None = None, probability_columns: Sequence[str] | None = None, threshold: float = 0.5, calibration_bins: int = 10) -> list[Path]:
    """Load aligned predictions and save all comparison figures in two formats."""
    if not input_csv.is_file():
        raise FileNotFoundError(f"Aligned prediction file not found: {input_csv}")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")

    frame = pd.read_csv(input_csv)
    detected_label = detect_label_column(frame, label_column)
    detected_probabilities = detect_probability_columns(frame, detected_label, probability_columns)
    predictions = validate_prediction_set(frame, detected_label, detected_probabilities)
    destination = output_directory or input_csv.parent / "figures"
    configure_style()

    figure_builders = (
        ("roc_curves", lambda: plot_roc(predictions)),
        ("precision_recall_curves", lambda: plot_precision_recall(predictions)),
        ("calibration_curves", lambda: plot_calibration(predictions, calibration_bins)),
        ("confusion_matrices", lambda: plot_confusion_matrices(predictions, threshold)),
        ("probability_distributions", lambda: plot_probability_distributions(predictions)),
        ("reliability_diagram", lambda: plot_reliability_diagram(predictions, calibration_bins)),
    )
    paths: list[Path] = []
    for stem, build_figure in figure_builders:
        png_path, pdf_path = save_figure(build_figure(), destination, stem)
        paths.extend((png_path, pdf_path))
    return paths


def main() -> None:
    """Run figure generation from the command line."""
    args = parse_args()
    paths = generate_figures(
        input_csv=args.input_csv,
        output_directory=args.output_directory,
        label_column=args.label_column,
        probability_columns=args.probability_columns,
        threshold=args.threshold,
        calibration_bins=args.calibration_bins,
    )
    print(f"Generated {len(paths)} files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
