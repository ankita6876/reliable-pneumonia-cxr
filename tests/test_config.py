"""Tests for environment-based dataset path configuration."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.config import DatasetPaths  # noqa: E402


def test_loads_configuration_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured temporary directories are loaded as Path objects."""
    chexpert = tmp_path / "chexpert"
    mimic = tmp_path / "mimic-cxr"
    nih = tmp_path / "nih-cxr"
    for dataset_path in (chexpert, mimic, nih):
        dataset_path.mkdir()

    monkeypatch.setenv("CHEXPERT_ROOT", str(chexpert))
    monkeypatch.setenv("MIMIC_CXR_ROOT", str(mimic))
    monkeypatch.setenv("NIH_CXR_ROOT", str(nih))

    paths = DatasetPaths.from_environment()

    assert paths.chexpert_root == chexpert
    assert paths.mimic_cxr_root == mimic
    assert paths.nih_cxr_root == nih


def test_raises_clear_error_for_missing_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All missing required variables are identified in the error message."""
    monkeypatch.delenv("CHEXPERT_ROOT", raising=False)
    monkeypatch.delenv("MIMIC_CXR_ROOT", raising=False)
    monkeypatch.delenv("NIH_CXR_ROOT", raising=False)

    with pytest.raises(EnvironmentError, match="CHEXPERT_ROOT.*MIMIC_CXR_ROOT.*NIH_CXR_ROOT"):
        DatasetPaths.from_environment()


def test_existence_report_uses_configured_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report distinguishes temporary existing and nonexistent paths."""
    chexpert = tmp_path / "chexpert"
    nih = tmp_path / "nih-cxr"
    chexpert.mkdir()
    nih.mkdir()

    monkeypatch.setenv("CHEXPERT_ROOT", str(chexpert))
    monkeypatch.setenv("MIMIC_CXR_ROOT", str(tmp_path / "missing-mimic"))
    monkeypatch.setenv("NIH_CXR_ROOT", str(nih))

    report = DatasetPaths.from_environment().existence_report()

    assert report == {
        "CheXpert": True,
        "MIMIC-CXR": False,
        "NIH ChestX-ray14": True,
    }
