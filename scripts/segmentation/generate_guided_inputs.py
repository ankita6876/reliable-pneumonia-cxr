"""Materialise deterministic segmentation-guided classifier images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image  # noqa: E402
from pneumonia_ai.segmentation.cache import MaskCache  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse materialisation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Metadata or split CSV.")
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument(
        "--image-root", type=Path,
        help="Root for relative image paths (defaults to the CSV directory).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=[InputMode.HARD_MASKED.value, InputMode.LUNG_CROP.value], required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--classifier-image-size", type=int, required=True)
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    """Generate images and a manifest while retaining per-image failures."""
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    table = pd.read_csv(args.csv)
    if args.image_path_column not in table:
        raise ValueError(f"CSV lacks image path column {args.image_path_column!r}.")
    rows = table.iloc[: args.limit] if args.limit is not None else table
    segmenter = FrozenLungSegmenter(args.checkpoint, args.device)
    cache = MaskCache(args.cache_directory) if args.cache_directory else None
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for raw_path in rows[args.image_path_column]:
        source = Path(str(raw_path)).expanduser()
        if not source.is_absolute():
            source = (args.image_root or args.csv.parent) / source
        target = args.output_directory / _output_name(source)
        record: dict[str, object] = {"original_image_path": str(source), "generated_image_path": str(target), "mode": args.mode, "checkpoint_identifier": str(args.checkpoint.resolve()), "threshold": args.threshold, "output_dimensions": f"{args.classifier_image_size}x{args.classifier_image_size}", "processing_status": "ok"}
        try:
            if target.exists() and not args.overwrite:
                record["processing_status"] = "skipped_existing"
            else:
                with Image.open(source) as opened:
                    image = opened.copy()
                probability = _cached_probability(cache, source, segmenter, image, args.threshold)
                generated = prepare_classifier_image(image, args.mode, segmenter=segmenter, threshold=args.threshold, crop_padding=args.crop_padding, output_size=args.classifier_image_size, probability_mask=probability)
                generated.save(target)
        except Exception as error:  # retain a complete manifest for batch triage
            record["processing_status"] = f"failed: {error}"
        manifest.append(record)
    manifest_path = args.output_directory / "guided_inputs_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    successes = sum(str(row["processing_status"]).startswith(("ok", "skipped")) for row in manifest)
    print(f"Completed {successes}/{len(manifest)} images; manifest: {manifest_path}")


def _cached_probability(cache: MaskCache | None, source: Path, segmenter: FrozenLungSegmenter, image: Image.Image, threshold: float):
    if cache is None:
        return segmenter.predict_proba(image)
    key = cache.key(source, segmenter.checkpoint_path, threshold, segmenter.image_size)
    probability = cache.get(key)
    if probability is None:
        probability = segmenter.predict_proba(image)
        cache.set(key, probability)
    return probability


def _output_name(source: Path) -> str:
    """Return a collision-safe deterministic PNG filename."""
    digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    return f"{source.stem}-{digest}.png"


if __name__ == "__main__":
    main()
