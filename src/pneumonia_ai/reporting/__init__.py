"""Metadata-only experiment discovery, comparison, and publication reporting."""

from .registry import build_registry, compare_registry, generate_publication_artifacts

__all__ = ["build_registry", "compare_registry", "generate_publication_artifacts"]
