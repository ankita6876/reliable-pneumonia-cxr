"""Schema-driven PadChest manifest tests using synthetic metadata only."""

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
    root = tmp_path / "images"
    for name in ("a.png", "b.png", "c.png", "d.png"):
        root.mkdir(exist_ok=True); Image.new("L", (2, 2)).save(root / name)
    return pd.DataFrame({
        "image": ["a.png", "b.png", "c.png", "d.png"],
        "patient": ["p1", "p1", "p2", "p3"], "case": ["a", "b", "c", "d"],
        "labels": ["Pneumonia|Effusion", "Mass", None, "Pneumonia"],
        "view": ["PA", "AP", "PA", "Lateral"],
    }), root


def test_report_level_labels_and_abnormal_negative_policy(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    manifest, summary = adapter.build_manifest(metadata, root, image_path_column="image", label_column="labels",
        pneumonia_concepts=["Pneumonia"], patient_id_column="patient", case_id_column="case",
        projection_column="view", accepted_views=["PA", "AP"])
    assert manifest.set_index("case_id").loc["a", "binary_target"] == 1
    assert manifest.set_index("case_id").loc["b", "binary_target"] == 0
    assert summary["exclusions"]["missing_or_unparseable_label"] == 1
    assert summary["exclusions"]["excluded_view"] == 1
    assert manifest.patient_id.tolist() == ["p1", "p1"]


def test_boundary_label_parsing_and_view_filtering(tmp_path: Path) -> None:
    assert adapter.parse_label_value('["Pneumonia", "Mass"]', "|") == ["Pneumonia", "Mass"]
    assert adapter.parse_label_value("", "|") is None
    metadata, root = _metadata(tmp_path)
    with pytest.raises(ValueError, match="both pneumonia target classes"):
        adapter.build_manifest(metadata, root, image_path_column="image", label_column="labels", pneumonia_concepts=["Pneumonia"], projection_column="view", accepted_views=["PA"])


def test_duplicate_case_or_image_rejected(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    metadata.loc[1, "case"] = "a"
    with pytest.raises(ValueError, match="identifiers must be unique"):
        adapter.build_manifest(metadata, root, image_path_column="image", label_column="labels", pneumonia_concepts=["Pneumonia"], case_id_column="case")


def test_schema_audit_does_not_freeze_likely_labels(tmp_path: Path) -> None:
    metadata, root = _metadata(tmp_path)
    args = type("Args", (), {"image_path_column": "image", "label_column": "labels", "projection_column": "view", "label_delimiter": "|"})()
    audit = adapter.schema_audit(metadata, root, args)
    assert audit["candidate_pneumonia_related_labels"] == ["Pneumonia"]
    assert audit["missing_image_path_count"] == 0
