"""Attribution and error-analysis utilities for binary pneumonia models."""

from .core import (
    attribution_quality,
    generate_attribution,
    select_cases,
    target_layer_for_model,
)

__all__ = ["attribution_quality", "generate_attribution", "select_cases", "target_layer_for_model"]
