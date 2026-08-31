from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from pneumonia_ai.classification.segmentation_guided import (
    InputMode,
    prepare_classifier_image,
)
from scripts.classification.optimisation_config import load_config


def _constant_image(size: int = 64, value: int = 200) -> Image.Image:
    return Image.fromarray(
        np.full((size, size), value, dtype=np.uint8),
        mode="L",
    )


def _square_probability(size: int = 64) -> torch.Tensor:
    probability = torch.zeros((size, size), dtype=torch.float32)
    probability[24:40, 24:40] = 1.0
    return probability


def test_context_preserving_mode_is_registered():
    assert InputMode.CONTEXT_PRESERVING.value == "context_preserving"


def test_context_preserving_keeps_lung_and_margin_full_strength():
    image = _constant_image()
    probability = _square_probability()

    output = np.asarray(
        prepare_classifier_image(
            image,
            InputMode.CONTEXT_PRESERVING,
            probability_mask=probability,
            threshold=0.5,
            context_dilation_radius=12,
            context_feather_radius=8,
            context_background_factor=0.20,
        )
    )

    # Inside thresholded lung.
    assert output[32, 32] == 200

    # Immediately outside lung.
    assert output[32, 41] == 200

    # Within the predefined 12-pixel full-context margin.
    assert output[32, 50] == 200


def test_context_preserving_far_background_is_twenty_percent():
    image = _constant_image()
    probability = _square_probability()

    output = np.asarray(
        prepare_classifier_image(
            image,
            "context_preserving",
            probability_mask=probability,
            context_dilation_radius=12,
            context_feather_radius=8,
            context_background_factor=0.20,
        )
    )

    # Corner is far beyond dilation + feather.
    assert output[0, 0] == 40


def test_context_preserving_feather_is_smooth_and_bounded():
    image = _constant_image()
    probability = _square_probability()

    output = np.asarray(
        prepare_classifier_image(
            image,
            "context_preserving",
            probability_mask=probability,
            context_dilation_radius=12,
            context_feather_radius=8,
            context_background_factor=0.20,
        )
    )

    # Moving horizontally away from the right lung edge:
    # x=50 is inside the full-strength context margin.
    # x=52 and x=55 are in the feather.
    # x=60 is distant background.
    values = [
        int(output[32, x])
        for x in (50, 52, 55, 60)
    ]

    assert values[0] == 200
    assert 40 < values[1] < 200
    assert 40 < values[2] < values[1]
    assert values[3] == 40


def test_context_preserving_output_shape_and_range():
    image = _constant_image()
    probability = _square_probability()

    output = np.asarray(
        prepare_classifier_image(
            image,
            "context_preserving",
            probability_mask=probability,
            output_size=224,
        )
    )

    assert output.shape == (224, 224)
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255


def test_historical_hard_mask_behavior_is_unchanged():
    image = Image.fromarray(
        np.array(
            [
                [100, 120],
                [140, 160],
            ],
            dtype=np.uint8,
        ),
        mode="L",
    )

    probability = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    output = np.asarray(
        prepare_classifier_image(
            image,
            "hard_masked",
            probability_mask=probability,
            threshold=0.5,
        )
    )

    expected = np.array(
        [
            [100, 0],
            [0, 160],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(output, expected)


def test_historical_soft_mask_behavior_is_unchanged():
    image = Image.fromarray(
        np.array(
            [
                [100, 120],
                [140, 160],
            ],
            dtype=np.uint8,
        ),
        mode="L",
    )

    probability = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    output = np.asarray(
        prepare_classifier_image(
            image,
            "soft_masked",
            probability_mask=probability,
            threshold=0.5,
            soft_mask_outside_factor=0.20,
        )
    )

    expected = np.array(
        [
            [100, 24],
            [28, 160],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "parameter,value,match",
    [
        ("context_dilation_radius", -1, "context_dilation_radius"),
        ("context_feather_radius", -1, "context_feather_radius"),
        ("context_background_factor", -0.1, "context_background_factor"),
        ("context_background_factor", 1.1, "context_background_factor"),
    ],
)
def test_context_parameters_reject_invalid_values(parameter, value, match):
    kwargs = {
        parameter: value,
        "probability_mask": torch.ones((2, 2)),
    }

    with pytest.raises(ValueError, match=match):
        prepare_classifier_image(
            Image.new("L", (2, 2), 100),
            "context_preserving",
            **kwargs,
        )


def test_context_config_loads_predefined_values(tmp_path: Path):
    path = tmp_path / "context.yaml"

    path.write_text(
        """
experiment: test_context
input_mode: context_preserving
mask_threshold: 0.5
context_dilation_radius: 12
context_feather_radius: 8
context_background_factor: 0.20
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.input_mode == "context_preserving"
    assert config.mask_threshold == 0.5
    assert config.context_dilation_radius == 12
    assert config.context_feather_radius == 8
    assert config.context_background_factor == 0.20


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_dilation_radius", 11),
        ("context_dilation_radius", 21),
        ("context_feather_radius", 7),
        ("context_background_factor", 0.30),
    ],
)


def test_context_config_refuses_unplanned_parameter_sweep(
    tmp_path: Path,
    field,
    value,
):
    path = tmp_path / "context.yaml"

    payload = {
        "experiment": "invalid_context",
        "input_mode": "context_preserving",
        "mask_threshold": 0.5,
        "context_dilation_radius": 12,
        "context_feather_radius": 8,
        "context_background_factor": 0.20,
    }

    payload[field] = value

    path.write_text(
        "\n".join(
            f"{key}: {str(val).lower() if isinstance(val, bool) else val}"
            for key, val in payload.items()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(path)

def test_context_config_accepts_predefined_twenty_pixel_fallback(tmp_path: Path):
    path = tmp_path / "context_r20.yaml"

    path.write_text(
        """
experiment: test_context_r20
input_mode: context_preserving
mask_threshold: 0.5
context_dilation_radius: 20
context_feather_radius: 8
context_background_factor: 0.20
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.input_mode == "context_preserving"
    assert config.context_dilation_radius == 20
    assert config.context_feather_radius == 8
    assert config.context_background_factor == 0.20


def test_context_radius_is_measured_after_resize_to_classifier_resolution():
    # Deliberately use a 448x448 source image and a 224x224 output.
    # If the radius were mistakenly measured at source resolution,
    # 12 source pixels would become only ~6 target pixels.
    #
    # This test therefore verifies that 12 means 12 pixels in the
    # final 224x224 classifier coordinate system.

    source_size = 448

    image = Image.fromarray(
        np.full(
            (source_size, source_size),
            200,
            dtype=np.uint8,
        )
    )

    probability = torch.zeros(
        (source_size, source_size),
        dtype=torch.float32,
    )

    # Central lung-like square.
    probability[128:320, 128:320] = 1.0

    output = prepare_classifier_image(
        image,
        mode=InputMode.CONTEXT_PRESERVING,
        probability_mask=probability,
        threshold=0.5,
        output_size=(224, 224),
        context_dilation_radius=12,
        context_feather_radius=8,
        context_background_factor=0.20,
    )

    array = np.asarray(output)

    assert array.shape == (224, 224)

    # After 448 -> 224 resize, left lung boundary is near x=64.
    y = 112

    # 12 target pixels outside boundary: should still be full strength.
    full_context_value = int(array[y, 52])

    # Around 16 target pixels outside: should lie inside feather.
    feather_value = int(array[y, 48])

    # >20 target pixels outside: should be near 20% of 200 = 40.
    far_background_value = int(array[y, 40])

    assert full_context_value >= 190

    assert 40 < feather_value < 190

    assert 35 <= far_background_value <= 45
