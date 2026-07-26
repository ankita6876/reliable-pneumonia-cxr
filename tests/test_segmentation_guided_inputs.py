from __future__ import annotations

from PIL import Image
import pytest
import torch

from pneumonia_ai.classification.segmentation_guided import prepare_classifier_image


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
    with pytest.raises(ValueError, match="mode"):
        prepare_classifier_image(image, "invalid")
