"""Validate CheXpert train and valid metadata without loading image pixels."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_metadata import (  # noqa: E402
    MetadataValidationError,
    SplitMetadataReport,
    validate_chexpert_metadata,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the CheXpert root directory.")
    return parser.parse_args()


def _print_split_report(split_name: str, report: SplitMetadataReport) -> None:
    """Print one split's readable validation summary."""
    print(f"{split_name} CSV: {report.csv_path}")
    print(f"  Rows: {report.row_count}")
    print(f"  Unique patients: {report.unique_patient_count}")
    print(f"  Frontal images: {report.frontal_image_count}")
    print(f"  Lateral images: {report.lateral_image_count}")
    print("  Pneumonia labels:")
    for label, count in report.pneumonia_label_counts.items():
        print(f"    {label}: {count}")
    print(f"  Missing image paths: {report.missing_image_path_count}")
    print(f"  Duplicate metadata rows: {report.duplicate_metadata_row_count}")


def main() -> int:
    """Print metadata validation results and return an appropriate status."""
    args = parse_args()
    try:
        validation = validate_chexpert_metadata(args.root)
    except (FileNotFoundError, MetadataValidationError, pd.errors.ParserError) as error:
        print(f"Metadata validation error: {error}", file=sys.stderr)
        return 1

    print(f"Root path: {validation.root_path}")
    _print_split_report("Train", validation.train)
    _print_split_report("Valid", validation.valid)
    print(f"Overlapping train/valid patients: {len(validation.overlapping_patient_ids)}")
    for patient_id in sorted(validation.overlapping_patient_ids):
        print(f"  {patient_id}")
    return 1 if validation.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
