"""CPU-only synthetic tests for attribution selection and normalized maps."""
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pneumonia_ai.explainability.core import generate_attribution, select_cases, target_layer_for_model  # noqa: E402


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.conv(value)).flatten(1))


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({"patient_id":["a","b","c","d"],"study_id":["1","2","3","4"],"image_path":["a.png","b.png","c.png","d.png"],"original_label":[0,1,0,1],"binary_target":[0,1,0,1],"logit":[-2,2,2,-2],"probability":[.1,.9,.9,.1],"predicted_class":[0,1,1,0],"split":["validation"]*4,"model_name":["toy"]*4,"run_id":["run"]*4})


def test_all_attribution_methods_return_normalized_input_shape_maps() -> None:
    model = _ToyModel()
    image = torch.rand(1, 8, 8)
    assert target_layer_for_model(model) is model.conv
    for method in ("gradcam", "gradcam++", "integrated_gradients", "occlusion"):
        result = generate_attribution(model, image, method)
        assert result.shape == (8, 8)
        assert 0 <= result.min() <= result.max() <= 1


def test_case_selection_includes_outcomes_and_uncertainty_extremes() -> None:
    selected = select_cases(_predictions(), .5, 1)
    assert {"true_positive", "true_negative", "false_positive", "false_negative", "highest_uncertainty", "lowest_uncertainty"} <= set(selected["outcome_group"])
