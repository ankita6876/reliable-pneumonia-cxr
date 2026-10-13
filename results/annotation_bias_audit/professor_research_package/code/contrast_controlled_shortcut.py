
import numpy as np

from shortcut_protocol import marker_position, stable_seed


def contrast_controlled_shortcut(
    rgb,
    lung_probability,
    label,
    image_id,
    region,
    condition,
    correlation=1.0,
    size=24,
    amplitude=45,
):
    """
    Deterministic, locally contrast-normalized marker.

    Positive label: vertical stripe pattern.
    Negative label: horizontal stripe pattern.

    Both patterns use the same two intensity levels.
    """

    image = np.asarray(rgb, dtype=np.uint8).copy()

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
            label if rng.random() < correlation else 1-label
        )
    else:
        raise ValueError("Unknown condition")

    y, x = position

    patch = image[y:y+size, x:x+size, :]
    background = float(patch.mean())

    # Keep the intended intensity separation while avoiding
    # clipping near the 0/255 boundaries.
    center = np.clip(
        background,
        amplitude,
        255-amplitude
    )

    low = int(round(center-amplitude))
    high = int(round(center+amplitude))

    yy, xx = np.indices((size, size))

    if marker_label == 1:
        pattern = (xx % 4) < 2
    else:
        pattern = (yy % 4) < 2

    marker = np.where(pattern, high, low).astype(np.uint8)

    image[y:y+size, x:x+size, :] = marker[..., None]

    return image, {
        "marker": "vertical" if marker_label == 1 else "horizontal",
        "position": [int(y), int(x)],
        "marker_label": int(marker_label),
        "low": low,
        "high": high,
    }
