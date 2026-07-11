"""Create deterministic patient-level splits from a CheXpert cohort CSV."""

import argparse
import sys
from pathlib import Path

from pandas.errors import ParserError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.splitting import (  # noqa: E402
    SPLIT_NAMES,
    CheXpertSplits,
    SplitValidationError,
    split_chexpert_cohort,
    write_chexpert_splits,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, help="Path to the cohort CSV.")
    parser.add_argument("--output", required=True, help="Path for the split cohort CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def print_report(splits: CheXpertSplits) -> None:
    """Print split distributions and integrity checks."""
    for split_name in SPLIT_NAMES:
        report = splits.reports[split_name]
        print(f"{split_name}:")
        print(f"  Row count: {report.row_count}")
        print(f"  Unique patient count: {report.unique_patient_count}")
        print(f"  Positive: {report.positive_count} ({report.positive_percentage:.1f}%)")
        print(f"  Negative: {report.negative_count} ({report.negative_percentage:.1f}%)")
        print(f"  Uncertain: {report.uncertain_count} ({report.uncertain_percentage:.1f}%)")
        print(f"  AP count: {report.ap_count}")
        print(f"  PA count: {report.pa_count}")
    print("Overlapping patients:")
    for pair, count in splits.overlapping_patient_counts.items():
        print(f"  {pair}: {count}")
    print(f"Total rows before splitting: {splits.rows_before}")
    print(f"Total rows after splitting: {splits.rows_after}")
    print(f"Missing rows: {splits.missing_row_count}")
    print(f"Duplicated assignments: {splits.duplicate_assignment_count}")


def main() -> int:
    """Create and save splits, returning non-zero on integrity failures."""
    args = parse_args()
    try:
        splits = split_chexpert_cohort(args.cohort, seed=args.seed)
    except (FileNotFoundError, ParserError, SplitValidationError) as error:
        print(f"Split error: {error}", file=sys.stderr)
        return 1

    print_report(splits)
    write_chexpert_splits(splits, args.output)
    print(f"Split CSV written: {args.output}")
    return 1 if splits.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
