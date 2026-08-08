"""Synthetic tests for the schema-frozen PadChest preparation adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "data" / "prepare_padchest_external.py"
SPEC = importlib.util.spec_from_file_location("prepare_padchest_external", PATH)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def _metadata(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    root = tmp_path / "images" / "images_001"
    root.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png", "d.png", "e.png", "f.png"):
        Image.new("L", (2, 2)).save(root / name)
    return pd.DataFrame({
        "ImageID": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
        "ImageDir": ["images_001"] * 6,
        "PatientID": ["p1", "p1", "p2", "p3", "p4", "p5"],
        "Projection": ["PA", "AP", "PA", "AP_horizontal", "L", "COSTAL"],
        "MethodLabel": ["R", "R", "M", "M", "R", "R"],
        "Labels": ["[' Pneumonia ' , 'Effusion']", "['Mass']", "['pneumonia-like']", None, "['PNEUMONIA']", "not-a-list"],
        "labelCUIS": ["['C0032285']", "['C002']", "['C003']", "[]", "['C0032285']", "[]"],
    }), root.parent


def test_exact_normalized_pneumonia_policy_and_missing_exclusion(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    manifest, summary = adapter.build_manifest(
        metadata, root, image_path_column=None, image_id_column="ImageID", image_dir_column="ImageDir",
        label_column="Labels", pneumonia_concepts=["pneumonia"], patient_id_column="PatientID",
        case_id_column="ImageID", projection_column="Projection", accepted_views=["PA", "AP", "AP_horizontal"],
    )
    targets = manifest.set_index("case_id").binary_target.to_dict()
    assert targets == {"a.png": 1, "b.png": 0, "c.png": 0}
    assert summary["exclusions"]["missing_or_unparseable_label"] == 2
    assert summary["exclusions"]["excluded_projection"] == 1
    assert "label_cuis" in manifest and manifest.loc[manifest.case_id.eq("a.png"), "label_cuis"].item() == "['C0032285']"


def test_python_like_lists_and_no_substring_matching() -> None:
    assert adapter.parse_label_value("[' Pneumonia ', 'Mass']") == ["pneumonia", "mass"]
    assert adapter.parse_label_value("['pneumonia-like']") == ["pneumonia-like"]
    assert adapter.parse_label_value("['pneumonia'") is None
    assert adapter.parse_label_value(None) is None


def test_configurable_projection_filtering_and_cohort_summary(tmp_path: Path) -> None:
    metadata, _ = _metadata(tmp_path)
    cohort, summary = adapter.build_cohort_request(
        metadata, image_id_column="ImageID", image_dir_column="ImageDir", patient_id_column="PatientID",
        projection_column="Projection", label_column="Labels", method_label_column="MethodLabel",
        accepted_views=["PA", "AP"],
    )
    assert cohort.ImageID.tolist() == ["a.png", "b.png", "c.png"]
    assert summary["cohort"] == {"total_images": 3, "unique_patients": 2, "pneumonia_positives": 1,
                                 "pneumonia_negatives": 2, "positive_prevalence": pytest.approx(1 / 3)}


def test_full_metadata_audit_counts_and_candidate_cohorts(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    args = type("Args", (), {"label_column": "Labels", "projection_column": "Projection", "patient_id_column": "PatientID", "method_label_column": "MethodLabel", "label_cuis_column": "labelCUIS", "image_id_column": "ImageID", "image_dir_column": "ImageDir", "image_path_column": None})()
    audit = adapter.schema_audit(metadata, root, args)
    assert audit["total_rows"] == 6 and audit["unique_images"] == 6 and audit["unique_patients"] == 5
    assert audit["valid_parsed_labels"] == 4 and audit["excluded_missing_or_unparseable_labels"] == 2
    assert audit["exact_pneumonia_positive_images_before_view_filtering"] == 2
    assert audit["normalized_labels_containing_pneum_audit_only"] == ["pneumonia", "pneumonia-like"]
    assert audit["label_cuis_on_exact_pneumonia_rows"] == ["['C0032285']"]
    assert audit["candidate_frontal_cohorts"]["PA + AP"]["pneumonia_positives"] == 1


def test_duplicate_image_id_rejected_for_cohort_request(tmp_path: Path) -> None:
    metadata, _ = _metadata(tmp_path)
    metadata.loc[1, "ImageID"] = "a.png"
    with pytest.raises(ValueError, match="duplicate ImageID"):
        adapter.build_cohort_request(metadata, image_id_column="ImageID", image_dir_column="ImageDir", patient_id_column="PatientID", projection_column="Projection", label_column="Labels")
