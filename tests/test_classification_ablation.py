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
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset

from scripts.classification.compare_ablations import build_comparison
from scripts.classification import train_ablation
from scripts.classification.evaluate_ablation import _resolve_input_mode


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


def test_evaluation_uses_explicit_input_mode() -> None:
    assert _resolve_input_mode({"input_mode": "lung_crop"}).value == "lung_crop"


def test_evaluation_uses_hard_masked_mode_for_legacy_checkpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _resolve_input_mode({}).value == "hard_masked"
    assert "WARNING" in capsys.readouterr().out


def test_evaluation_rejects_invalid_explicit_input_mode() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        _resolve_input_mode({"input_mode": "masked"})
