"""Safe, reusable conversion of chest-radiograph DICOMs to RGB PIL images."""

from pathlib import Path

import numpy as np
from PIL import Image


def dicom_to_pil(path: Path | str) -> Image.Image:
    """Read a DICOM and return an 8-bit RGB image without changing model transforms."""
    import pydicom

    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        pixels = -pixels
    lower, upper = np.percentile(pixels, (1, 99))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(pixels.min()), float(pixels.max())
    if upper <= lower:
        scaled = np.zeros_like(pixels, dtype=np.uint8)
    else:
        scaled = np.clip((pixels - lower) / (upper - lower), 0, 1)
        scaled = np.round(scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")
