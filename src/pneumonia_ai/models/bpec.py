"""Boundary-preserving exterior perturbation consistency.

Training-only anatomical perturbation. The inference model remains
an ordinary full-image classifier.
"""
import torch
import torch.nn.functional as F


def exterior_eligibility(lung_probability, threshold=0.5, radius=12):
    """Return pixels safely outside a dilated lung region."""
    if lung_probability.ndim == 3:
        lung_probability = lung_probability.unsqueeze(1)
    if lung_probability.ndim != 4 or lung_probability.shape[1] != 1:
        raise ValueError("Expected Bx1xHxW lung probabilities")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be between zero and one")
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    lung = (lung_probability >= threshold).float()
    kernel = 2 * radius + 1
    protected = F.max_pool2d(
        lung, kernel_size=kernel, stride=1, padding=radius
    )
    return (protected == 0).float()


def perturb_exterior(
    images,
    lung_probability,
    patch_size=8,
    patches_per_image=2,
    boundary_radius=12,
    threshold=0.5,
    fill_value=0.0,
):
    """Replace complete exterior patches; preserve all other pixels.

    Input images are ImageNet-normalized. fill_value=0 replaces
    selected pixels with the channel-wise normalized value zero,
    corresponding to ImageNet channel means before normalization.

    Returns perturbed images and a Bx1xHxW selected-pixel mask.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW images")
    if patch_size < 1 or patches_per_image < 0:
        raise ValueError("Invalid patch configuration")

    b, _, h, w = images.shape
    if patch_size > min(h, w):
        raise ValueError("Patch larger than image")

    if lung_probability.shape[-2:] != (h, w):
        lung_probability = F.interpolate(
            lung_probability.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

    eligible = exterior_eligibility(
        lung_probability,
        threshold=threshold,
        radius=boundary_radius,
    ).to(images.device)

    selected = torch.zeros_like(eligible)

    for i in range(b):
        candidates = F.avg_pool2d(
            eligible[i:i+1],
            kernel_size=patch_size,
            stride=1,
        )[0, 0]

        coords = torch.nonzero(candidates >= 1.0 - 1e-6)
        if coords.numel() == 0:
            continue

        order = torch.randperm(
            len(coords), device=coords.device
        )

        count = 0
        for idx in order:
            y, x = coords[idx].tolist()
            if selected[i, 0, y:y+patch_size, x:x+patch_size].any():
                continue
            selected[i, 0, y:y+patch_size, x:x+patch_size] = 1.0
            count += 1
            if count >= patches_per_image:
                break

    perturbed = images * (1.0 - selected) + fill_value * selected
    return perturbed, selected


def bpec_consistency_loss(original_logits, perturbed_logits):
    """Mean squared probability difference."""
    original = torch.sigmoid(original_logits.float())
    perturbed = torch.sigmoid(perturbed_logits.float())
    return ((original - perturbed) ** 2).mean()
