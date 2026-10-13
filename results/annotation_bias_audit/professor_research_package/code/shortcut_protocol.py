"""Reproducible label-correlated artificial shortcut protocol.

Marker:
  positive label -> bright square
  negative label -> dark square

Training correlation:
  p(correct marker | label) = 0.90

Evaluation:
  clean, matching, reversed

Anatomical locations:
  interior, boundary, exterior

The same location is used for all marker conditions
within an image, avoiding location-induced confounding.
"""
import hashlib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def stable_seed(*values):
    message = "|".join(map(str, values))
    digest = hashlib.sha256(message.encode()).digest()
    return int.from_bytes(digest[:8], "little")


def anatomical_regions(lung_probability, radius=12):
    lung = np.asarray(lung_probability).squeeze() >= 0.5
    if lung.ndim != 2:
        raise ValueError("Expected a 2D lung probability map")

    dilated = binary_dilation(lung, iterations=radius)
    eroded = binary_erosion(lung, iterations=radius)

    return {
        "interior": eroded,
        "boundary": dilated & ~eroded,
        "exterior": ~dilated,
    }


def eligible_positions(region_mask, size=12):
    mask = np.asarray(region_mask, dtype=np.int32)
    integral = np.pad(
        mask, ((1, 0), (1, 0))
    ).cumsum(0).cumsum(1)

    counts = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return np.argwhere(counts == size * size)


def marker_position(lung_probability, region, image_id, size=12):
    regions = anatomical_regions(lung_probability)
    positions = eligible_positions(regions[region], size)

    if len(positions) == 0:
        return None

    rng = np.random.default_rng(
        stable_seed("position", image_id, region)
    )
    y, x = positions[rng.integers(len(positions))]
    return int(y), int(x)


def shortcut_image(
    rgb,
    lung_probability,
    label,
    image_id,
    region,
    condition,
    correlation=0.90,
    size=12,
):
    """Return modified RGB image and metadata.

    condition:
        clean    -> no marker
        train    -> label-correlated marker
        matching -> always label-matching marker
        reversed -> always opposite-label marker
    """
    image = np.asarray(rgb, dtype=np.uint8).copy()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image")

    if label not in (0, 1):
        raise ValueError("Only binary labels supported")

    if condition == "clean":
        return image, {"marker": "none", "position": None}

    position = marker_position(
        lung_probability, region, image_id, size
    )

    if position is None:
        return image, {
            "marker": "unavailable",
            "position": None
        }

    if condition == "matching":
        marker_label = label

    elif condition == "reversed":
        marker_label = 1 - label

    elif condition == "train":
        rng = np.random.default_rng(
            stable_seed("association", image_id)
        )
        marker_label = (
            label if rng.random() < correlation else 1 - label
        )

    else:
        raise ValueError("Unknown condition")

    y, x = position
    value = 230 if marker_label == 1 else 25

    image[y:y + size, x:x + size, :] = value

    return image, {
        "marker": "bright" if marker_label == 1 else "dark",
        "position": [y, x],
        "marker_label": int(marker_label),
    }


def self_test():
    image = np.full((224, 224, 3), 120, dtype=np.uint8)
    yy, xx = np.ogrid[:224, :224]

    lung = (
        (((xx - 75) / 35)**2 + ((yy - 115) / 65)**2 < 1)
        |
        (((xx - 150) / 35)**2 + ((yy - 115) / 65)**2 < 1)
    )

    for region in ("interior", "boundary", "exterior"):
        results = {}

        for condition in ("clean", "train", "matching", "reversed"):
            changed, meta = shortcut_image(
                image, lung, 1, "synthetic_001",
                region, condition
            )
            results[condition] = (changed, meta)

        clean, clean_meta = results["clean"]
        matching, matching_meta = results["matching"]
        reversed_img, reversed_meta = results["reversed"]

        assert np.array_equal(clean, image)
        assert matching_meta["marker"] == "bright"
        assert reversed_meta["marker"] == "dark"
        assert matching_meta["position"] == reversed_meta["position"]

        region_mask = anatomical_regions(lung)[region]
        y, x = matching_meta["position"]
        assert region_mask[y:y+12, x:x+12].all()

        assert np.count_nonzero(
            np.any(matching != image, axis=2)
        ) == 144

        assert np.count_nonzero(
            np.any(reversed_img != image, axis=2)
        ) == 144

        print(region, "PASSED")

    # Confirm deterministic marker assignment.
    a, ma = shortcut_image(
        image, lung, 1, "test", "exterior", "train"
    )
    b, mb = shortcut_image(
        image, lung, 1, "test", "exterior", "train"
    )
    assert np.array_equal(a, b)
    assert ma == mb

    print("ALL SHORTCUT PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    self_test()
