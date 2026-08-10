from __future__ import annotations

from PIL import Image
import pytest
import torch
import numpy as np

from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image


class _Segmenter:
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask
        self.calls = 0

    def predict_proba(self, image: Image.Image) -> torch.Tensor:
        self.calls += 1
        return self.mask


def test_hard_mask_crop_padding_empty_fallback_and_original() -> None:
    image = Image.fromarray(__import__("numpy").arange(36, dtype="uint8").reshape(6, 6))
    mask = torch.zeros(6, 6)
    mask[2:4, 1:3] = 1
    segmenter = _Segmenter(mask)
    masked = prepare_classifier_image(image, "hard_masked", segmenter=segmenter)
    cropped = prepare_classifier_image(image, "lung_crop", segmenter=segmenter, crop_padding=1, output_size=4)
    original = prepare_classifier_image(image, "original", output_size=4)
    assert masked.getpixel((0, 0)) == 0 and masked.getpixel((1, 2)) == image.getpixel((1, 2))
    assert cropped.size == original.size == (4, 4)
    empty = prepare_classifier_image(image, "lung_crop", segmenter=_Segmenter(torch.zeros(6, 6)))
    assert list(empty.getdata()) == list(image.getdata())


def test_guided_validation_and_original_never_calls_segmenter() -> None:
    image = Image.new("L", (3, 3))
    segmenter = _Segmenter(torch.ones(3, 3))
    prepare_classifier_image(image, "original", segmenter=segmenter)
    assert segmenter.calls == 0
    with pytest.raises(ValueError, match="requires"):
        prepare_classifier_image(image, "hard_masked")
    with pytest.raises(ValueError, match="requires"):
        prepare_classifier_image(image, "soft_masked")
    with pytest.raises(ValueError, match="mode"):
        prepare_classifier_image(image, "invalid")


def test_soft_masked_uses_the_hard_mask_threshold_and_deterministic_uint8_attenuation() -> None:
    image = Image.fromarray(np.array([[1, 4, 10], [20, 99, 255]], dtype=np.uint8))
    probability = torch.tensor([[.5, .49, 1.0], [.0, .5, .1]])
    soft = np.asarray(prepare_classifier_image(image, InputMode.SOFT_MASKED, probability_mask=probability))
    original = np.asarray(image)
    mask = probability.numpy() >= .5
    assert np.array_equal(soft[mask], original[mask])
    assert np.array_equal(soft[~mask], (original[~mask].astype(np.float32) * .20).astype(np.uint8))


def test_soft_mask_extremes_match_hard_masked_and_original() -> None:
    image = Image.fromarray(np.arange(36, dtype=np.uint8).reshape(6, 6))
    probability = torch.tensor([[1, 0, 1, 0, 1, 0]] * 6, dtype=torch.float32)
    hard = prepare_classifier_image(image, "hard_masked", probability_mask=probability, output_size=4)
    zero = prepare_classifier_image(image, "soft_masked", probability_mask=probability, soft_mask_outside_factor=0, output_size=4)
    original = prepare_classifier_image(image, "original", output_size=4)
    one = prepare_classifier_image(image, "soft_masked", probability_mask=probability, soft_mask_outside_factor=1, output_size=4)
    assert np.array_equal(np.asarray(zero), np.asarray(hard))
    assert np.array_equal(np.asarray(one), np.asarray(original))


@pytest.mark.parametrize("factor", (-.01, 1.01))
def test_soft_mask_factor_must_be_a_unit_interval(factor: float) -> None:
    with pytest.raises(ValueError, match="soft_mask_outside_factor"):
        prepare_classifier_image(Image.new("L", (2, 2)), "soft_masked", probability_mask=torch.ones(2, 2), soft_mask_outside_factor=factor)
