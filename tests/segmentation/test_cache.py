from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from pneumonia_ai.segmentation.cache import MaskCache


def test_cache_hit_miss_invalidation_and_corruption_recovery(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "x.png", tmp_path / "model.pt"
    Image.new("L", (4, 3), color=10).save(source)
    checkpoint.write_bytes(b"checkpoint")
    cache = MaskCache(tmp_path / "cache")
    key = cache.key(source, checkpoint, 0.5, 8)
    assert cache.get(key) is None
    cache.set(key, torch.ones(3, 4))
    assert torch.equal(cache.get(key), torch.ones(3, 4))
    assert cache.key(source, checkpoint, 0.6, 8) != key
    (cache.directory / f"{key}.npz").write_bytes(b"corrupt")
    assert cache.get(key) is None
