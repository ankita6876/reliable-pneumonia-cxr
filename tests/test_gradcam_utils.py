from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.analysis.gradcam_utils import (
    SelectedCase,
    activation_localisation,
    save_case_figure,
    select_representative_cases,
)


def test_case_selection_prefers_high_confidence_examples() -> None:
    predictions = pd.DataFrame({
        "image_path": ["tp_low", "tp_high", "tn_low", "tn_high", "fp_low", "fp_high", "fn_low", "fn_high"],
        "binary_target": [1, 1, 0, 0, 0, 0, 1, 1],
        "predicted_class": [1, 1, 0, 0, 1, 1, 0, 0],
        "probability": [0.6, 0.9, 0.2, 0.1, 0.6, 0.95, 0.4, 0.05],
    })

    cases = select_representative_cases(predictions, cases_per_category=1)

    assert [(case.category, case.image_path) for case in cases] == [
        ("TP", "tp_high"), ("TN", "tn_high"), ("FP", "fp_high"), ("FN", "fn_high"),
    ]


def test_activation_localisation_partitions_activation_mass() -> None:
    values = activation_localisation(np.array([[1.0, 3.0], [0.0, 0.0]]), np.array([[True, False], [True, False]]))

    assert values["activation_inside_lungs"] == 0.25
    assert values["activation_outside_lungs"] == 0.75
    assert values["lung_focus_ratio"] == 1 / 3


def test_case_figure_writes_requested_format(tmp_path: Path) -> None:
    path = tmp_path / "case.png"
    pdf_path = tmp_path / "case.pdf"
    case = SelectedCase("TP", "image.png", 1, 1, 0.9, "patient", "study")

    save_case_figure(
        path,
        Image.fromarray(np.full((12, 12), 100, dtype=np.uint8)),
        np.ones((12, 12), dtype=bool),
        Image.fromarray(np.full((12, 12), 100, dtype=np.uint8)),
        np.ones((8, 8), dtype=float),
        case,
    )
    save_case_figure(
        pdf_path,
        Image.fromarray(np.full((12, 12), 100, dtype=np.uint8)),
        np.ones((12, 12), dtype=bool),
        Image.fromarray(np.full((12, 12), 100, dtype=np.uint8)),
        np.ones((8, 8), dtype=float),
        case,
    )

    assert path.is_file()
    assert pdf_path.is_file()
