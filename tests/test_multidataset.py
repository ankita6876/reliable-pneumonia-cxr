"""Synthetic metadata tests for the unified multi-dataset manifest interface."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.multidataset import STANDARD_COLUMNS, build_standard_manifest  # noqa: E402


def _image(root: Path, path: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (2, 2)).save(target)


def test_chexpert_nih_and_mimic_share_a_standard_manifest(tmp_path: Path) -> None:
    root = tmp_path / "images"
    sources = {
        "chexpert": ("CheXpert-v1.0-small/train/patient1/study1/view.jpg", {"Path": ["CheXpert-v1.0-small/train/patient1/study1/view.jpg"], "Pneumonia": [1]}),
        "nih": ("00000001_000.png", {"Image Index": ["00000001_000.png"], "Finding Labels": ["Pneumonia"], "Patient ID": [1]}),
        "mimic": ("files/p12/p123/s4/dicom.jpg", {"subject_id": [123], "study_id": [4], "dicom_id": ["dicom"], "Pneumonia": [0]}),
    }
    for dataset, (image_path, metadata) in sources.items():
        _image(root, image_path)
        csv_path = tmp_path / f"{dataset}.csv"
        pd.DataFrame(metadata).to_csv(csv_path, index=False)
        manifest = build_standard_manifest(dataset, csv_path, root)
        assert tuple(manifest.columns) == STANDARD_COLUMNS
        assert manifest.loc[0, "dataset"] == dataset
        assert manifest.loc[0, "split"] == "external"
        assert not Path(manifest.loc[0, "image_path"]).is_absolute()
