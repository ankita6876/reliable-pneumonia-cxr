from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from pneumonia_ai.classification.ablation import (
    AblationConfig,
    _development_manifest,
    _validate_config,
)
from pneumonia_ai.data.chexpert_dataset import (
    CheXpertPneumoniaDataset,
    DatasetValidationError,
    _resolve_manifest_image_path,
)

from scripts.classification.compare_ablations import build_comparison
from scripts.classification import train_ablation
from scripts.classification.evaluate_ablation import normalize_checkpoint_configuration


def _config(tmp_path: Path, mode: str, checkpoint: Path | None = None) -> AblationConfig:
    return AblationConfig(
        input_mode=mode,
        splits_csv=tmp_path / "chexpert_splits.csv",
        image_root=tmp_path,
        output_directory=tmp_path / "output",
        segmentation_checkpoint=checkpoint,
    )


def test_ablation_mode_validation_and_segmenter_requirement(tmp_path: Path) -> None:
    assert _validate_config(_config(tmp_path, "original")).value == "original"
    with pytest.raises(ValueError, match="require"):
        _validate_config(_config(tmp_path, "hard_masked"))
    with pytest.raises(ValueError, match="input_mode"):
        _validate_config(_config(tmp_path, "not-a-mode"))


def test_comparison_csv_has_all_modes_and_metrics(tmp_path: Path) -> None:
    values = {
        "auc": 0.8, "pr_auc": 0.7, "f1": 0.6, "ece": 0.1,
        "brier": 0.2, "accuracy": 0.75, "recall": 0.7, "specificity": 0.8,
    }
    for mode in ("original", "hard_masked", "lung_crop"):
        directory = tmp_path / mode
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(values))

    comparison = build_comparison(tmp_path)

    assert comparison["mode"].tolist() == ["original", "hard_masked", "lung_crop"]
    assert pd.read_csv(tmp_path / "comparison.csv").columns.tolist() == ["mode", *values]


def test_development_manifest_excludes_test_split(tmp_path: Path) -> None:
    splits_csv = tmp_path / "chexpert_splits.csv"
    pd.DataFrame(
        {
            "image_path": ["train.png", "validation.png", "test.png"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(splits_csv, index=False)

    manifest = _development_manifest(splits_csv, tmp_path)

    assert pd.read_csv(manifest)["split"].tolist() == ["train", "validation"]


def test_ablation_image_root_resolves_relative_and_permits_absolute_paths(tmp_path: Path) -> None:
    image_root = tmp_path / "extracted"
    relative_image = image_root / "train" / "patient" / "view.jpg"
    relative_image.parent.mkdir(parents=True)
    Image.new("L", (4, 4)).save(relative_image)
    absolute_image = tmp_path / "outside.jpg"
    Image.new("L", (4, 4)).save(absolute_image)
    manifest = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "patient_id": ["one", "two"], "study_id": ["one", "two"],
            "image_path": [r"train\patient\view.jpg", str(absolute_image)],
            "pneumonia_label": [0, 1], "split": ["train", "validation"],
        }
    ).to_csv(manifest, index=False)

    train = CheXpertPneumoniaDataset(image_root, manifest, "train")
    validation = CheXpertPneumoniaDataset(
        image_root, manifest, "validation", allow_absolute_image_paths=True
    )

    assert train._resolved_image_paths == [relative_image]
    assert validation._resolved_image_paths == [absolute_image]


def test_manifest_image_path_resolution_is_portable_and_safe(tmp_path: Path) -> None:
    image_root = tmp_path / "extracted"
    image_root.mkdir()
    absolute_image = tmp_path / "outside.jpg"

    assert _resolve_manifest_image_path(image_root, r"train\patient\view.jpg") == (
        image_root / "train" / "patient" / "view.jpg"
    )
    assert _resolve_manifest_image_path(image_root, "train/patient/view.jpg") == (
        image_root / "train" / "patient" / "view.jpg"
    )
    assert _resolve_manifest_image_path(
        image_root, absolute_image, allow_absolute=True
    ) == absolute_image.resolve()
    with pytest.raises(DatasetValidationError, match="relative POSIX path"):
        _resolve_manifest_image_path(image_root, absolute_image)
    with pytest.raises(DatasetValidationError, match="relative POSIX path"):
        _resolve_manifest_image_path(image_root, r"train\..\outside.jpg")


def test_ablation_image_root_reports_missing_resolved_image(tmp_path: Path) -> None:
    manifest = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "patient_id": ["one"], "study_id": ["one"],
            "image_path": ["train/missing.jpg"], "pneumonia_label": [0], "split": ["train"],
        }
    ).to_csv(manifest, index=False)

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        CheXpertPneumoniaDataset(tmp_path, manifest, "train")


def test_training_cli_smoke_parses_original_without_segmenter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ablation.py", "--input-mode", "original", "--splits-csv",
            str(tmp_path / "chexpert_splits.csv"),
            "--image-root", str(tmp_path),
            "--output-directory", str(tmp_path / "output"),
        ],
    )
    args = train_ablation.parse_args()
    assert args.input_mode == "original"
    assert args.splits_csv == tmp_path / "chexpert_splits.csv"
    assert args.image_root == tmp_path
    assert args.segmentation_checkpoint is None


def test_evaluation_normalizes_modern_checkpoint_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration = {
        "input_mode": "lung_crop", "classifier_image_size": 256,
        "mask_threshold": 0.7, "lung_crop_padding": 3,
    }

    normalized = normalize_checkpoint_configuration(configuration)

    assert normalized == configuration
    assert normalized is not configuration
    assert not capsys.readouterr().out


def test_evaluation_normalizes_missing_legacy_input_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = normalize_checkpoint_configuration(
        {"classifier_image_size": 224, "mask_threshold": 0.5, "lung_crop_padding": 0}
    )

    assert normalized["input_mode"] == "hard_masked"
    assert "input_mode" in capsys.readouterr().out


def test_evaluation_normalizes_legacy_input_size_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = normalize_checkpoint_configuration(
        {"input_mode": "hard_masked", "input_size": 224, "mask_threshold": 0.5,
         "lung_crop_padding": 0}
    )

    assert normalized["classifier_image_size"] == 224
    assert "checkpoint input_size" in capsys.readouterr().out


def test_evaluation_uses_adjacent_configuration_for_missing_legacy_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = normalize_checkpoint_configuration(
        {"input_mode": "hard_masked"},
        {"input_size": 256, "mask_threshold": 0.6, "lung_crop_padding": 2},
    )

    assert normalized["classifier_image_size"] == 256
    assert normalized["mask_threshold"] == 0.6
    assert normalized["lung_crop_padding"] == 2
    assert capsys.readouterr().out.count("adjacent config.json") == 3


def test_evaluation_normalizes_multiple_legacy_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = normalize_checkpoint_configuration({})

    assert normalized == {
        "input_mode": "hard_masked", "classifier_image_size": 224,
        "mask_threshold": 0.5, "lung_crop_padding": 0,
    }
    assert capsys.readouterr().out.count("WARNING") == 4


def test_evaluation_preserves_checkpoint_values_over_adjacent_configuration() -> None:
    normalized = normalize_checkpoint_configuration(
        {"input_mode": "original", "classifier_image_size": 320, "mask_threshold": 0.6,
         "lung_crop_padding": 4},
        {"input_mode": "hard_masked", "input_size": 224, "mask_threshold": 0.5,
         "lung_crop_padding": 0},
    )

    assert normalized["input_mode"] == "original"
    assert normalized["classifier_image_size"] == 320
    assert normalized["mask_threshold"] == 0.6
    assert normalized["lung_crop_padding"] == 4


def test_evaluation_reports_all_unresolved_configuration_fields() -> None:
    with pytest.raises(ValueError, match=(
        "input_mode.*classifier_image_size.*mask_threshold.*lung_crop_padding"
    )):
        normalize_checkpoint_configuration(
            {"input_mode": None, "classifier_image_size": None,
             "mask_threshold": None, "lung_crop_padding": None}
        )
