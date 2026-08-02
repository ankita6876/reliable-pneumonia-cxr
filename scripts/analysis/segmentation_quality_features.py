"""Deterministic, interpretable quality descriptors for soft lung masks."""
from __future__ import annotations

from typing import Mapping
import numpy as np


QUALITY_FEATURE_COLUMNS = (
    "mask_probability_mean", "mask_probability_std", "mask_entropy_mean", "mask_entropy_std",
    "uncertain_pixel_fraction_0_4_0_6", "uncertain_pixel_fraction_0_25_0_75",
    "foreground_area_ratio", "foreground_pixel_count", "connected_component_count",
    "largest_component_area_ratio", "second_largest_component_area_ratio", "component_area_ratio_sum_top2",
    "border_touch_fraction", "mask_centroid_x_normalized", "mask_centroid_y_normalized",
    "bounding_box_width_ratio", "bounding_box_height_ratio", "bounding_box_area_ratio",
    "boundary_length_normalized", "boundary_roughness", "left_area_ratio", "right_area_ratio",
    "left_right_area_symmetry", "left_right_centroid_vertical_difference", "left_right_component_balance",
)
FLAG_COLUMNS = ("empty_mask", "single_side_mask", "touches_image_border")


def binary_entropy(probability: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Stable binary entropy in nats."""
    p = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def segmentation_quality_features(probability_mask: np.ndarray, threshold: float = .5) -> dict[str, float | int | bool]:
    """Return finite geometric descriptors; undefined side-only metrics are explicit NaN."""
    p = np.asarray(probability_mask, dtype=float)
    if p.ndim != 2 or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("Probability mask must be a non-empty finite 2-D array.")
    if not 0 < threshold < 1: raise ValueError("threshold must be strictly between 0 and 1.")
    p = np.clip(p, 0, 1); mask = p >= threshold; h, w = mask.shape; n = h * w
    entropy = binary_entropy(p); ys, xs = np.where(mask); count = len(xs); empty = count == 0
    components = _components(mask); areas = sorted((len(c) for c in components), reverse=True)
    top1, top2 = (areas + [0, 0])[:2]
    border = np.zeros_like(mask, bool); border[[0, -1], :] = True; border[:, [0, -1]] = True
    left, right = mask[:, :w // 2], mask[:, w // 2:]
    la, ra = int(left.sum()), int(right.sum())
    one_side = not empty and (la == 0 or ra == 0)
    boundary = _boundary_length(mask)
    result: dict[str, float | int | bool] = {
        "mask_probability_mean": float(p.mean()), "mask_probability_std": float(p.std()),
        "mask_entropy_mean": float(entropy.mean()), "mask_entropy_std": float(entropy.std()),
        "uncertain_pixel_fraction_0_4_0_6": float(((p >= .4) & (p <= .6)).mean()),
        "uncertain_pixel_fraction_0_25_0_75": float(((p >= .25) & (p <= .75)).mean()),
        "foreground_area_ratio": count / n, "foreground_pixel_count": count,
        "connected_component_count": len(components), "largest_component_area_ratio": top1 / n,
        "second_largest_component_area_ratio": top2 / n, "component_area_ratio_sum_top2": (top1 + top2) / n,
        "border_touch_fraction": float((mask & border).sum() / count) if count else 0.,
        "boundary_length_normalized": boundary / max(h + w, 1),
        "boundary_roughness": boundary / max(2 * np.sqrt(np.pi * count), 1),
        "left_area_ratio": la / n, "right_area_ratio": ra / n,
        "left_right_area_symmetry": 1 - abs(la - ra) / (la + ra) if la + ra else np.nan,
        "left_right_centroid_vertical_difference": np.nan,
        "left_right_component_balance": min(la, ra) / max(la, ra) if la and ra else np.nan,
        "empty_mask": empty, "single_side_mask": one_side, "touches_image_border": bool((mask & border).any()),
    }
    if empty:
        result.update({"mask_centroid_x_normalized": np.nan, "mask_centroid_y_normalized": np.nan,
                       "bounding_box_width_ratio": 0., "bounding_box_height_ratio": 0., "bounding_box_area_ratio": 0.})
    else:
        result.update({"mask_centroid_x_normalized": float(xs.mean() / max(w - 1, 1)), "mask_centroid_y_normalized": float(ys.mean() / max(h - 1, 1)),
                       "bounding_box_width_ratio": (xs.max() - xs.min() + 1) / w, "bounding_box_height_ratio": (ys.max() - ys.min() + 1) / h,
                       "bounding_box_area_ratio": ((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))/n})
    if la and ra:
        result["left_right_centroid_vertical_difference"] = abs(np.where(left)[0].mean() - np.where(right)[0].mean()) / max(h - 1, 1)
    return result


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components without adding a scipy dependency."""
    seen = np.zeros_like(mask, bool); h, w = mask.shape; found = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]: continue
        stack, component = [(int(y), int(x))], []; seen[y, x] = True
        while stack:
            cy, cx = stack.pop(); component.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx] = True; stack.append((ny,nx))
        found.append(component)
    return found


def _boundary_length(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(int), 1)
    return float(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
