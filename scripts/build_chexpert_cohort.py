"""Build the unsplit eligible CheXpert pneumonia cohort."""

import argparse
import sys
from pathlib import Path

from pandas.errors import ParserError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_cohort import (  # noqa: E402
    CheXpertCohort,
    build_chexpert_cohort,
    write_chexpert_cohort,
)
from pneumonia_ai.data.chexpert_metadata import MetadataValidationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the CheXpert root directory.")
    parser.add_argument("--output", help="Optional CSV path for eligible cohort records.")
    return parser.parse_args()


def print_report(cohort: CheXpertCohort) -> None:
    """Print the requested cohort-construction report."""
    print(f"Total eligible rows: {cohort.total_eligible_rows}")
    print(f"Unique patients: {cohort.unique_patients}")
    print(f"Positive count: {cohort.positive_count}")
    print(f"Negative count: {cohort.negative_count}")
    print(f"Uncertain count: {cohort.uncertain_count}")
    print(f"AP count: {cohort.ap_count}")
    print(f"PA count: {cohort.pa_count}")
    print(f"Missing AP/PA count: {cohort.missing_ap_pa_count}")
    print(f"Excluded lateral rows: {cohort.excluded_lateral_rows}")
    print(f"Excluded missing-label rows: {cohort.excluded_missing_label_rows}")
    print(f"Missing image paths: {cohort.missing_image_path_count}")
    print(f"Duplicate rows: {cohort.duplicate_row_count}")
    print(f"Duplicate image paths: {cohort.duplicate_image_path_count}")
    print(f"Conflicting-label image paths: {cohort.conflicting_label_image_path_count}")


def main() -> int:
    """Build the cohort and return an appropriate process exit code."""
    args = parse_args()
    try:
        cohort = build_chexpert_cohort(args.root)
    except (FileNotFoundError, MetadataValidationError, ParserError) as error:
        print(f"Cohort build error: {error}", file=sys.stderr)
        return 1

    print_report(cohort)
    if args.output:
        write_chexpert_cohort(cohort, args.output)
        print(f"Cohort CSV written: {args.output}")
    return 1 if cohort.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
