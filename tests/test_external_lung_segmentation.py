"""Focused tests for the inference-only JSRT/SCR segmentation evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analysis" / "evaluate_external_lung_segmentation.py"
SPEC = importlib.util.spec_from_file_location("external_lung_segmentation", SCRIPT_PATH)
assert SPEC and SPEC.loader
external = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = external
SPEC.loader.exec_module(external)


def _write_gray(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype("uint8")).save(path)


def _mapped_case(tmp_path: Path, case_id: str = "JPCLN001") -> tuple[Path, Path]:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _write_gray(images / f"{case_id}.png", np.full((4, 5), 100, dtype=np.uint8))
    _write_gray(masks / "left" / f"{case_id}.png", np.array([[0, 1, 0, 0, 0]] * 4))
    _write_gray(masks / "right" / f"{case_id}.png", np.array([[0, 0, 0, 1, 0]] * 4))
    return images, masks


def _combined_case(tmp_path: Path, case_id: str = "JPCNN040") -> tuple[Path, Path]:
    images = tmp_path / "cxr"
    masks = tmp_path / "combined_masks"
    _write_gray(images / f"{case_id}.png", np.full((4, 5), 100, dtype=np.uint8))
    _write_gray(masks / f"{case_id}.png", np.array([[0, 127, 128, 255, 0]] * 4))
    return images, masks


def test_merge_metrics_and_empty_mask_semantics() -> None:
    merged = external.merge_masks_for_case
    # The merge operation itself is exercised with temporary image files below.
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    truth = np.array([[1, 0], [0, 0]], dtype=bool)
    dice, iou = external.binary_dice_iou(prediction, truth)
    assert dice == pytest.approx(2 / 3)
    assert iou == pytest.approx(1 / 2)
    assert external.binary_dice_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == (1.0, 1.0)
    assert external.binary_dice_iou(np.ones((2, 2), bool), np.zeros((2, 2), bool)) == (0.0, 0.0)
    assert callable(merged)


def test_left_right_mask_merge(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    _write_gray(left, np.array([[0, 255], [0, 0]], dtype=np.uint8))
    _write_gray(right, np.array([[0, 0], [9, 0]], dtype=np.uint8))
    assert external.merge_masks_for_case(left, right).tolist() == [[False, True], [True, False]]


def test_combined_grayscale_mask_uses_explicit_128_decoding_threshold(tmp_path: Path) -> None:
    mask = tmp_path / "JPCNN040.png"
    _write_gray(mask, np.array([[0, 127, 128, 255]], dtype=np.uint8))
    assert external.binarize_combined_grayscale_mask(mask).tolist() == [[False, False, True, True]]
    assert external.COMBINED_GT_MASK_THRESHOLD == 128
    assert external.MASK_THRESHOLD == 0.5


def test_bootstrap_and_sample_selection_are_deterministic() -> None:
    values = np.array([0.1, 0.4, 0.9])
    assert external.bootstrap_summary(values, 30, 42) == external.bootstrap_summary(values, 30, 42)
    mappings = [external.CaseMapping(f"JPCLN{i:03d}", None, None, None, "mapped") for i in range(1, 10)]
    assert external.deterministic_sample(mappings, 4, 42) == external.deterministic_sample(mappings, 4, 42)


def test_checkpoint_hash_verification_rejects_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-test-checkpoint")
    observed = external.sha256_file(checkpoint)
    assert external.verify_checkpoint_sha256(checkpoint, observed) == observed
    with pytest.raises(ValueError, match="mismatch"):
        external.verify_checkpoint_sha256(checkpoint, "0" * 64)


def test_mapping_duplicate_rejection_and_missing_masks(tmp_path: Path) -> None:
    images, masks = _mapped_case(tmp_path)
    _write_gray(images / "duplicate_JPCLN001.png", np.zeros((4, 5), dtype=np.uint8))
    mappings = external.discover_mappings(images, masks)
    assert mappings[0].mapping_status == "ambiguous"
    with pytest.raises(ValueError, match="Unresolved"):
        external.require_resolved_mappings(mappings)

    other = tmp_path / "other"
    images, masks = _mapped_case(other)
    (masks / "right" / "JPCLN001.png").unlink()
    mappings = external.discover_mappings(images, masks)
    assert mappings[0].mapping_status == "missing_file"
    assert "missing right mask" in mappings[0].notes


def test_combined_mapping_pairing_missing_and_duplicate_safeguards(tmp_path: Path) -> None:
    images, masks = _combined_case(tmp_path)
    mappings = external.discover_mappings(images, None, combined_masks_root=masks)
    assert len(mappings) == 1
    assert mappings[0].mapping_status == "mapped"
    assert mappings[0].case_id == "JPCNN040"
    assert mappings[0].combined_mask_path == masks / "JPCNN040.png"

    missing = tmp_path / "missing"
    images, masks = _combined_case(missing)
    (masks / "JPCNN040.png").unlink()
    mappings = external.discover_mappings(images, None, combined_masks_root=masks)
    assert mappings[0].mapping_status == "missing_file"
    assert "missing combined mask" in mappings[0].notes
    with pytest.raises(ValueError, match="Unresolved"):
        external.require_resolved_mappings(mappings)

    duplicate = tmp_path / "duplicate"
    images, masks = _combined_case(duplicate)
    _write_gray(masks / "copy_JPCNN040.png", np.zeros((4, 5), dtype=np.uint8))
    mappings = external.discover_mappings(images, None, combined_masks_root=masks)
    assert mappings[0].mapping_status == "ambiguous"


class _Segmenter:
    def __init__(self) -> None:
        self.metadata = {
            "model_config": {"in_channels": 1, "out_channels": 1, "base_channels": 16, "depth": 3},
            "training_config": {"image_size": 128},
        }
        self.image_size = 128

    def predict_proba(self, image: Image.Image) -> torch.Tensor:
        result = torch.zeros(image.height, image.width)
        result[:, 1] = 0.8
        result[:, 3] = 0.9
        return result


def test_shape_mismatch_is_explicit_failure(tmp_path: Path) -> None:
    images, masks = _mapped_case(tmp_path)
    _write_gray(masks / "right" / "JPCLN001.png", np.zeros((3, 5), dtype=np.uint8))
    mapping = external.discover_mappings(images, masks)
    metrics, failures = external.evaluate_mappings(mapping, _Segmenter(), "abc")
    assert metrics.empty
    assert len(failures) == 1
    assert "different resolutions" in failures.loc[0, "failure"]


def test_combined_mask_geometry_mismatch_is_explicit_failure(tmp_path: Path) -> None:
    images, masks = _combined_case(tmp_path)
    _write_gray(masks / "JPCNN040.png", np.zeros((3, 5), dtype=np.uint8))
    mappings = external.discover_mappings(images, None, combined_masks_root=masks)
    metrics, failures = external.evaluate_mappings(
        mappings, _Segmenter(), "abc", ground_truth_mask_mode="combined_grayscale"
    )
    assert metrics.empty
    assert "Image/mask shape mismatch" in failures.loc[0, "failure"]


def test_run_writes_metadata_and_keeps_postprocessing_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images, masks = _mapped_case(tmp_path)
    output = tmp_path / "out"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(external, "verify_checkpoint_sha256", lambda path, expected: "a" * 64)
    monkeypatch.setattr(external, "FrozenLungSegmenter", lambda path, device: _Segmenter())
    args = Namespace(images_root=images, masks_root=masks, left_masks_root=None, right_masks_root=None, combined_masks_root=None,
                     mapping_csv=None, checkpoint=checkpoint, output_dir=output, device="cpu",
                     bootstrap_iterations=20, seed=42, max_samples=None,
                     expected_checkpoint_sha256="a" * 64, audit_only=False, overwrite=False)
    metadata = external.run(args)
    assert metadata["mask_threshold"] == 0.5
    assert metadata["postprocessing"] == "none"
    assert (output / "segmentation_mapping_audit.csv").is_file()
    assert (output / "segmentation_qualitative_panel.png").is_file()
    saved = json.loads((output / "segmentation_metadata.json").read_text())
    assert saved["number_evaluated"] == 1
    assert saved["checkpoint_sha256"] == "a" * 64


def test_combined_run_metadata_distinguishes_ground_truth_and_prediction_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images, masks = _combined_case(tmp_path)
    output = tmp_path / "out"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(external, "verify_checkpoint_sha256", lambda path, expected: "a" * 64)
    monkeypatch.setattr(external, "FrozenLungSegmenter", lambda path, device: _Segmenter())
    args = Namespace(images_root=images, masks_root=None, left_masks_root=None, right_masks_root=None,
                     combined_masks_root=masks, mapping_csv=None, checkpoint=checkpoint, output_dir=output,
                     device="cpu", bootstrap_iterations=20, seed=42, max_samples=None,
                     expected_checkpoint_sha256="a" * 64, audit_only=False, overwrite=False)
    metadata = external.run(args)
    summary = json.loads((output / "segmentation_summary.json").read_text())
    assert metadata["ground_truth_mask_mode"] == "combined_grayscale"
    assert metadata["ground_truth_mask_threshold"] == 128
    assert metadata["prediction_mask_threshold"] == 0.5
    assert metadata["postprocessing"] == "none"
    assert summary["ground_truth_mask_binarization"] == "uint8 >= 128"
    assert summary["mask_threshold"] == 0.5
    assert summary["prediction_mask_threshold"] == 0.5
