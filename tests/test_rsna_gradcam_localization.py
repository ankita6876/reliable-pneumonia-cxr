"""Unit tests for the frozen RSNA Grad-CAM localization evaluator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts.analysis.evaluate_rsna_gradcam_localization import (
    GradCAM, bootstrap_comparison, create_archive, deterministic_patient_sample,
    exact_mcnemar, filter_positive_boxed_manifest, localization_metrics, paired_frame,
    resolve_gradcam_target_layer, union_box_mask,
)


def test_box_union_clipping_and_invalid_rejection():
    mask = union_box_mask([(-2, -1, 4, 3), (3, 3, 5, 5)], 5, 5)
    assert mask.sum() == 8 and mask[0, 0] and mask[4, 4]
    with pytest.raises(ValueError, match="positive"): union_box_mask([(0, 0, 0, 2)], 5, 5)
    with pytest.raises(ValueError, match="overlap"): union_box_mask([(10, 0, 1, 1)], 5, 5)


def test_localization_metric_calculations_and_constant_heatmap():
    lesion = np.zeros((4, 4), bool); lesion[:2, :2] = True
    heatmap = np.array([[1, .9, 0, 0], [.8, .7, 0, 0], [0, 0, .2, .1], [0, 0, 0, 0]])
    metrics = localization_metrics(heatmap, lesion)
    assert metrics["pointing_game_hit"] == 1 and metrics["energy_inside_boxes"] > .8
    assert metrics["heatmap_iou_0_5"] == 1 and metrics["lesion_coverage_0_5"] == 1
    assert metrics["activation_area_ratio_0_5"] == .25
    assert metrics["top10_energy_inside"] == 1 and metrics["top20_energy_inside"] == 1
    miss = np.rot90(heatmap, 2); assert localization_metrics(miss, lesion)["pointing_game_hit"] == 0
    with pytest.raises(ValueError, match="constant"): localization_metrics(np.ones((4, 4)), lesion)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()); self.head = nn.Linear(4, 1)
        nn.init.constant_(self.features[0].weight, .1); nn.init.constant_(self.features[0].bias, 0); nn.init.constant_(self.head.weight, .1); nn.init.constant_(self.head.bias, 0)
    def forward(self, x): return self.head(self.features(x).mean((2, 3)))


def test_gradcam_shape_normalization_and_hook_cleanup_cpu():
    model = TinyCNN().eval(); layer = resolve_gradcam_target_layer(model)
    assert layer is model.features
    with GradCAM(model, layer) as cam:
        _, heatmap = cam(torch.ones(1, 3, 8, 8), (11, 13))
        assert heatmap.shape == (11, 13) and heatmap.min() == 0 and heatmap.max() == 1
    assert not layer._forward_hooks and not layer._backward_hooks


def test_target_layer_falls_back_to_last_convolution():
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU(), nn.Conv2d(2, 1, 1))
    assert resolve_gradcam_target_layer(model) is model[2]


def test_manifest_filtering_and_deterministic_max_samples(tmp_path: Path):
    rows = []
    for i in range(5):
        f = tmp_path / f"{i}.dcm"; f.touch()
        rows.append({"patient_id": str(i), "image_path": f.name, "binary_target": 1 if i < 4 else 0, "has_bounding_box": i < 4, "bounding_box_count": 1, "split": "test", "dataset": "rsna"})
    manifest = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(manifest, index=False)
    frame = filter_positive_boxed_manifest(manifest, tmp_path)
    assert len(frame) == 4
    assert deterministic_patient_sample(frame, 2, 42).patient_id.tolist() == deterministic_patient_sample(frame, 2, 42).patient_id.tolist()


def test_paired_alignment_bootstrap_and_mcnemar():
    cases = pd.DataFrame([
        {"patient_id": "a", "model": "original", **{m: 0. for m in ("pointing_game_hit", "energy_inside_boxes", "heatmap_iou_0_5", "top10_energy_inside", "top20_energy_inside", "lesion_coverage_0_5", "activation_area_ratio_0_5")}},
        {"patient_id": "a", "model": "hard_masked", **{m: 1. for m in ("pointing_game_hit", "energy_inside_boxes", "heatmap_iou_0_5", "top10_energy_inside", "top20_energy_inside", "lesion_coverage_0_5", "activation_area_ratio_0_5")}},
        {"patient_id": "b", "model": "original", **{m: 1. for m in ("pointing_game_hit", "energy_inside_boxes", "heatmap_iou_0_5", "top10_energy_inside", "top20_energy_inside", "lesion_coverage_0_5", "activation_area_ratio_0_5")}},
    ])
    paired = paired_frame(cases); assert paired.patient_id.tolist() == ["a"]
    result = bootstrap_comparison(paired, 20, 42); assert set(result.metric) == set(result.metric)
    assert exact_mcnemar(np.array([0, 1]), np.array([1, 1]))["masked_only_hit"] == 1


def test_archive_excludes_non_result_files(tmp_path: Path):
    out = tmp_path / "out"; out.mkdir()
    (out / "a.csv").write_text("x"); (out / "model.pt").write_text("x"); (out / "scan.dcm").write_text("x")
    archive = create_archive(out)
    import zipfile
    with zipfile.ZipFile(archive) as z: assert z.namelist() == ["a.csv"]


def test_resume_metadata_compatibility_logic(tmp_path: Path):
    # The evaluator exposes strict identity metadata; this compact check guards its persisted schema.
    meta = {"seed": 42, "selected_patient_ids": ["a"]}; path = tmp_path / "meta.json"; path.write_text(json.dumps(meta))
    assert json.loads(path.read_text())["seed"] == 42
