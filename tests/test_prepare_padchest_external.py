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
    manifest, missing, summary = adapter.build_manifest(
        metadata, root, image_path_column=None, image_id_column="ImageID", image_dir_column="ImageDir",
        label_column="Labels", pneumonia_concepts=["pneumonia"], patient_id_column="PatientID",
        case_id_column="ImageID", projection_column="Projection", accepted_views=["PA", "AP", "AP_horizontal"],
    )
    targets = manifest.set_index("case_id").binary_target.to_dict()
    assert targets == {"a.png": 1, "b.png": 0, "c.png": 0}
    assert manifest.image_path.tolist() == ["images_001/a.png", "images_001/b.png", "images_001/c.png"]
    assert summary["exclusions"]["missing_or_unparseable_label"] == 2
    assert summary["exclusions"]["excluded_projection"] == 1
    assert "label_cuis" in manifest and manifest.loc[manifest.case_id.eq("a.png"), "label_cuis"].item() == "['C0032285']"
    assert missing.empty and summary["intended_cohort_count"] == 3 and summary["available_evaluable_count"] == 3


def test_manifest_records_missing_frozen_cohort_image_with_provenance(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    (root / "images_001" / "b.png").unlink()
    manifest, missing, summary = adapter.build_manifest(
        metadata, root, image_path_column=None, image_id_column="ImageID", image_dir_column="ImageDir",
        label_column="Labels", pneumonia_concepts=["pneumonia"], patient_id_column="PatientID",
        case_id_column="ImageID", projection_column="Projection", method_label_column="MethodLabel",
        accepted_views=["PA", "AP"],
    )
    assert manifest.case_id.tolist() == ["a.png", "c.png"]
    assert missing[["case_id", "binary_target", "projection", "method_label", "exclusion_reason"]].to_dict("records") == [{"case_id": "b.png", "binary_target": 0, "projection": "AP", "method_label": "R", "exclusion_reason": "image_unavailable_in_image_root"}]
    assert summary["intended_cohort_count"] == 3 and summary["available_evaluable_count"] == 2 and summary["missing_image_count"] == 1


def test_flat_imageid_mirror_preserves_metadata_provenance(tmp_path: Path) -> None:
    metadata, _ = _metadata(tmp_path)
    flat_root = tmp_path / "flat"
    flat_root.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("L", (2, 2)).save(flat_root / name)
    manifest, missing, _ = adapter.build_manifest(
        metadata, flat_root, image_path_column=None, image_id_column="ImageID", image_dir_column="ImageDir",
        label_column="Labels", pneumonia_concepts=["pneumonia"], patient_id_column="PatientID",
        case_id_column="ImageID", projection_column="Projection", method_label_column="MethodLabel",
        accepted_views=["PA", "AP"],
    )
    assert manifest[["image_path", "patient_id", "binary_target", "projection", "image_dir", "method_label"]].to_dict("records") == [
        {"image_path": "a.png", "patient_id": "p1", "binary_target": 1, "projection": "PA", "image_dir": "images_001", "method_label": "R"},
        {"image_path": "b.png", "patient_id": "p1", "binary_target": 0, "projection": "AP", "image_dir": "images_001", "method_label": "R"},
        {"image_path": "c.png", "patient_id": "p2", "binary_target": 0, "projection": "PA", "image_dir": "images_001", "method_label": "M"},
    ]
    assert missing.empty
    args = type("Args", (), {"label_column": "Labels", "projection_column": "Projection", "patient_id_column": "PatientID", "method_label_column": "MethodLabel", "label_cuis_column": "labelCUIS", "image_id_column": "ImageID", "image_dir_column": "ImageDir", "image_path_column": None})()
    assert adapter.schema_audit(metadata, flat_root, args)["missing_image_path_count"] == 3


def test_image_resolver_does_not_fallback_to_an_unrelated_filename(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    Image.new("L", (2, 2)).save(root / "unrelated.png")
    record = pd.Series({"ImageID": "expected.png", "ImageDir": "images_001"})
    assert adapter.resolve_image_relative_path(record, root, image_path_column=None, image_id_column="ImageID", image_dir_column="ImageDir") is None


def test_python_like_lists_and_no_substring_matching() -> None:
    cases = {
        "[]": None,
        None: None,
        float("nan"): None,
        "['pneumonia'": None,
        "not-a-list": None,
        "['normal']": ["normal"],
        "['pneumonia']": ["pneumonia"],
        "[' pneumonia ']": ["pneumonia"],
        "['pneumonia-like']": ["pneumonia-like"],
    }
    for raw, expected in cases.items():
        assert adapter.parse_label_value(raw) == expected
    assert adapter.parse_label_value("[' Pneumonia ', 'Mass']") == ["pneumonia", "mass"]


def test_unusable_labels_are_excluded_and_valid_labels_receive_exact_targets() -> None:
    metadata = pd.DataFrame({"Labels": ["[]", None, float("nan"), "['pneumonia'", "['normal']", "['pneumonia']", "[' pneumonia ']", "['pneumonia-like']"]})
    labelled, exclusions = adapter._labelled_rows(metadata, label_column="Labels")
    assert exclusions == {"missing_or_unparseable_label": 4, "excluded_projection": 0}
    assert [row["pneumonia_target"] for row in labelled] == [0, 1, 1, 0]


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
