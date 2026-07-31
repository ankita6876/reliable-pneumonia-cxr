from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.plot_results import (
    detect_label_column,
    detect_probability_columns,
    generate_figures,
)


def test_detects_generic_label_and_probability_columns() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "binary_target": [0, 1, 0, 1],
            "probability_baseline": [0.1, 0.8, 0.2, 0.7],
            "probability_crop": [0.2, 0.9, 0.3, 0.6],
        }
    )

    label = detect_label_column(frame)

    assert label == "binary_target"
    assert detect_probability_columns(frame, label) == ["probability_baseline", "probability_crop"]


def test_generate_figures_writes_png_and_pdf_pairs(tmp_path: Path) -> None:
    input_csv = tmp_path / "aligned.csv"
    pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "probability_a": [0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6],
            "probability_b": [0.2, 0.7, 0.4, 0.8, 0.1, 0.6, 0.3, 0.9],
        }
    ).to_csv(input_csv, index=False)

    paths = generate_figures(input_csv, calibration_bins=4)

    assert len(paths) == 12
    assert all(path.is_file() for path in paths)
