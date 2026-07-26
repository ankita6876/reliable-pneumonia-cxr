from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pneumonia_ai.classification.ablation import AblationConfig, _validate_config

from scripts.classification.compare_ablations import build_comparison
from scripts.classification import train_ablation


def _config(tmp_path: Path, mode: str, checkpoint: Path | None = None) -> AblationConfig:
    return AblationConfig(
        input_mode=mode,
        train_csv=tmp_path / "train.csv",
        validation_csv=tmp_path / "validation.csv",
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


def test_training_cli_smoke_parses_original_without_segmenter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ablation.py", "--input-mode", "original", "--train-csv",
            str(tmp_path / "train.csv"), "--validation-csv", str(tmp_path / "validation.csv"),
            "--output-directory", str(tmp_path / "output"),
        ],
    )
    args = train_ablation.parse_args()
    assert args.input_mode == "original"
    assert args.segmentation_checkpoint is None
