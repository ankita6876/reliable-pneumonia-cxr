from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional
from torch.utils.data import TensorDataset

from pneumonia_ai.segmentation.losses import BCEWithLogitsDiceLoss, BinaryDiceLoss
from pneumonia_ai.segmentation.metrics import per_sample_metrics, segmentation_metrics
from pneumonia_ai.segmentation.model import UNet
from pneumonia_ai.segmentation.training import (
    build_dataloader,
    build_optimizer,
    seed_everything,
    train_one_epoch,
    validate_one_epoch,
)


def test_unet_output_shape_forward_and_backward() -> None:
    model = UNet(base_channels=4, depth=2)
    images = torch.randn(2, 1, 32, 32)
    logits = model(images)

    assert logits.shape == images.shape
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_unet_preserves_odd_spatial_dimensions() -> None:
    images = torch.randn(1, 1, 65, 67)

    logits = UNet(base_channels=4, depth=3)(images)

    assert logits.shape == images.shape


def test_dice_loss_distinguishes_perfect_and_incorrect_predictions() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    perfect_logits = torch.where(targets.bool(), torch.tensor(20.0), torch.tensor(-20.0))
    incorrect_logits = -perfect_logits
    loss = BinaryDiceLoss()

    assert loss(perfect_logits, targets).item() < 1e-4
    assert loss(incorrect_logits, targets).item() > 0.99


def test_combined_loss_respects_configured_weights() -> None:
    logits = torch.tensor([[[[0.4, -0.2]]]])
    targets = torch.tensor([[[[1.0, 0.0]]]])
    bce_only = BCEWithLogitsDiceLoss(bce_weight=1.0, dice_weight=0.0)
    combined = BCEWithLogitsDiceLoss(bce_weight=0.5, dice_weight=0.5)

    assert torch.allclose(bce_only(logits, targets), functional.binary_cross_entropy_with_logits(logits, targets))
    assert combined(logits, targets).item() > 0.0
    with pytest.raises(ValueError, match="equal shapes"):
        bce_only(logits, targets[..., :1])


def test_metrics_aggregate_per_sample_values_and_do_not_track_gradients() -> None:
    logits = torch.tensor(
        [[[[20.0, 20.0], [-20.0, -20.0]]], [[[-20.0, -20.0], [-20.0, -20.0]]]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]
    )

    values = per_sample_metrics(logits, targets)
    result = segmentation_metrics(logits, targets)

    assert values["dice"].requires_grad is False
    assert result.dice == pytest.approx(0.75)
    assert result.iou == pytest.approx((1.0 / 3.0 + 1.0) / 2.0)
    assert result.sensitivity == pytest.approx(0.75)
    assert result.specificity == pytest.approx(0.75)
    assert result.precision == pytest.approx(0.75)


def test_metrics_handle_empty_masks_explicitly() -> None:
    empty_targets = torch.zeros(1, 1, 2, 2)

    empty_prediction = segmentation_metrics(torch.full_like(empty_targets, -20.0), empty_targets)
    false_positive = segmentation_metrics(torch.full_like(empty_targets, 20.0), empty_targets)

    assert empty_prediction.dice == empty_prediction.iou == empty_prediction.precision == 1.0
    assert empty_prediction.sensitivity == empty_prediction.specificity == 1.0
    assert false_positive.dice == false_positive.iou == false_positive.precision == 0.0
    assert false_positive.sensitivity == 1.0
    assert false_positive.specificity == 0.0


def test_cpu_training_updates_parameters_and_validation_does_not() -> None:
    seed_everything(42)
    images = torch.rand(4, 1, 16, 16)
    masks = (images > 0.5).to(dtype=torch.float32)
    loader = build_dataloader(TensorDataset(images, masks), batch_size=2, shuffle=True)
    model = UNet(base_channels=4, depth=1)
    optimizer = build_optimizer(model, learning_rate=1e-2)
    loss = BCEWithLogitsDiceLoss()
    before_training = [parameter.detach().clone() for parameter in model.parameters()]

    summary = train_one_epoch(model, loader, optimizer, loss, gradient_clip_norm=1.0)

    assert summary.samples == 4
    assert math.isfinite(summary.loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_training, model.parameters())
    )
    before_validation = [parameter.detach().clone() for parameter in model.parameters()]
    validation = validate_one_epoch(model, loader, loss)
    assert validation.samples == 4
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_validation, model.parameters())
    )


def test_seeding_is_deterministic() -> None:
    seed_everything(42)
    first = torch.rand(4)
    seed_everything(42)
    second = torch.rand(4)

    assert torch.equal(first, second)
