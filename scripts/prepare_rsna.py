"""Prepare a portable RSNA external-validation manifest and cohort artifacts."""
# ruff: noqa: E501, E701, E702
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.data.rsna import build_rsna_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest, summary = build_rsna_manifest(args.labels, args.image_root, allow_missing_images=args.allow_missing_images)
    manifest.to_csv(output / "rsna_manifest.csv", index=False)
    pd.DataFrame([summary.as_dict()]).to_csv(output / "rsna_cohort_summary.csv", index=False)
    (output / "rsna_integrity_report.json").write_text(json.dumps(summary.as_dict(), indent=2))
    counts = manifest["label"].value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(["Negative", "Positive"], counts.to_numpy())
    total = len(manifest)
    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count} ({count / total:.1%})", ha="center", va="bottom")
    axis.set_title("RSNA external-validation class distribution")
    axis.set_ylabel("Patient count")
    figure.tight_layout(); figure.savefig(output / "rsna_class_distribution.png", dpi=300); plt.close(figure)


if __name__ == "__main__":
    main()
