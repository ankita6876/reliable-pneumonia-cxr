from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from scripts.analysis import generate_gradcam
from scripts.analysis.gradcam_utils import (
    GradCAM,
    SelectedCase,
    activation_localisation,
    save_case_figure,
    select_representative_cases,
)


def test_gradcam_parser_accepts_cpu_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both supported execution devices are accepted by the Grad-CAM CLI."""
    monkeypatch.setattr(sys, "argv", ["generate_gradcam.py", "--device", "cpu"])
    assert generate_gradcam.parse_args().device == "cpu"
    monkeypatch.setattr(sys, "argv", ["generate_gradcam.py", "--device", "cuda"])
    assert generate_gradcam.parse_args().device == "cuda"


def test_gradcam_cuda_preflight_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_gradcam.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"--device cuda was requested.*is_available\(\) is False"):
        generate_gradcam.generate_gradcam_analysis(
            classifier_checkpoint=Path("missing-classifier.pt"),
            predictions_csv=Path("missing-predictions.csv"),
            segmentation_checkpoint=Path("missing-segmenter.pt"),
            output_directory=Path("unused"),
            device="cuda",
        )


def test_classifier_loading_moves_model_to_selected_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU checkpoint loading is followed by model placement on the selected device."""
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.touch()

    class TrackingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Conv2d(1, 1, 1)
            self.received_device: torch.device | None = None

        def to(self, *args: object, **kwargs: object) -> "TrackingModel":
            self.received_device = torch.device(args[0])
            return super().to(*args, **kwargs)  # type: ignore[arg-type]

    model = TrackingModel()
    state = {
        "configuration": {"input_mode": "hard_masked", "model": "test-model"},
        "model_state_dict": model.state_dict(),
    }
    monkeypatch.setattr(generate_gradcam.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(generate_gradcam, "create_model", lambda *args, **kwargs: model)

    loaded, _ = generate_gradcam._load_classifier(checkpoint, torch.device("cpu"))

    assert loaded is model
    assert model.received_device == torch.device("cpu")
    assert next(model.parameters()).device == torch.device("cpu")


def test_gradcam_forwards_input_on_model_device() -> None:
    """Grad-CAM's forward/backward path preserves the classifier input device."""
    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.input_device: torch.device | None = None

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            self.input_device = image.device
            return self.conv(image).mean(dim=(2, 3))

    model = RecordingModel().to("cpu")
    cam = GradCAM(model, model.conv)
    try:
        _ = cam.generate(torch.ones(1, 1, 4, 4, device="cpu"), predicted_class=1)
    finally:
        cam.close()

    assert model.input_device == next(model.parameters()).device


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
