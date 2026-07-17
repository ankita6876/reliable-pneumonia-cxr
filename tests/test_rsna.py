"""Synthetic RSNA manifest and DICOM tests; no clinical images are used."""
# ruff: noqa: E501, E701, E702
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.dicom import dicom_to_pil  # noqa: E402
from pneumonia_ai.data.rsna import RSNAValidationError, build_rsna_manifest  # noqa: E402
from pneumonia_ai.evaluation.core import aggregate_ensemble, validate_predictions  # noqa: E402


def _labels(path: Path, values: list[tuple[str, int]]) -> Path:
    result = path / "labels.csv"; pd.DataFrame(values, columns=["patientId", "Target"]).to_csv(result, index=False); return result


def test_rsna_manifest_collapses_boxes_and_is_portable(tmp_path: Path) -> None:
    root = tmp_path / "images"; root.mkdir(); (root / "positive.dcm").touch(); (root / "negative.dcm").touch()
    manifest, summary = build_rsna_manifest(_labels(tmp_path, [("positive", 1), ("positive", 1), ("negative", 0)]), root)
    assert len(manifest) == 2 and manifest.set_index("patient_id").loc["positive", "label"] == 1
    assert manifest.set_index("patient_id").loc["negative", "label"] == 0
    assert summary.duplicate_annotation_rows == 1 and not Path(manifest.loc[0, "image_path"]).is_absolute()


def test_rsna_rejects_missing_conflicting_and_malformed_labels(tmp_path: Path) -> None:
    root = tmp_path / "images"; root.mkdir()
    with pytest.raises(FileNotFoundError): build_rsna_manifest(_labels(tmp_path, [("missing", 0)]), root)
    with pytest.raises(RSNAValidationError, match="conflicting"):
        build_rsna_manifest(_labels(tmp_path, [("same", 0), ("same", 1)]), root, allow_missing_images=True)
    with pytest.raises(RSNAValidationError, match="binary"):
        build_rsna_manifest(_labels(tmp_path, [("bad", 2)]), root, allow_missing_images=True)


def _dicom(path: Path, pixels: np.ndarray, photometric: str, slope: float = 1, intercept: float = 0) -> None:
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian; meta.MediaStorageSOPClassUID = generate_uid(); meta.MediaStorageSOPInstanceUID = generate_uid()
    data = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128); data.Rows, data.Columns = pixels.shape; data.SamplesPerPixel = 1; data.PhotometricInterpretation = photometric; data.BitsAllocated = data.BitsStored = 16; data.HighBit = 15; data.PixelRepresentation = 0; data.RescaleSlope = slope; data.RescaleIntercept = intercept; data.PixelData = pixels.astype(np.uint16).tobytes(); data.save_as(str(path))


def test_dicom_loader_handles_monochrome_and_rescale(tmp_path: Path) -> None:
    pixels = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    mono2, mono1 = tmp_path / "m2.dcm", tmp_path / "m1.dcm"; _dicom(mono2, pixels, "MONOCHROME2", 2, 10); _dicom(mono1, pixels, "MONOCHROME1")
    image2, image1 = np.asarray(dicom_to_pil(mono2)), np.asarray(dicom_to_pil(mono1))
    assert image2.shape == (2, 2, 3) and image2[0, 0, 0] < image2[1, 1, 0]
    assert image1[0, 0, 0] > image1[1, 1, 0]


def test_external_prediction_schema_and_identifier_aggregation() -> None:
    first = pd.DataFrame({"patient_id": ["a", "b"], "image_id": ["a", "b"], "image_path": ["a.dcm", "b.dcm"], "label": [0, 1], "probability": [.1, .9], "logit": [-2, 2], "split": "external_test", "dataset": "rsna", "model": "toy", "checkpoint": "one"})
    second = first.iloc[::-1].reset_index(drop=True).assign(probability=[.8, .2], logit=[1.4, -1.4], checkpoint="two")
    validate_predictions(first)
    ensemble = aggregate_ensemble([first, second])
    assert {"probability", "predictive_entropy", "variance", "mutual_information", "binary_disagreement"} <= set(ensemble)
