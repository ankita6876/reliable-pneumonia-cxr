"""Validate configured dataset-root paths without accessing image files."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.config import DatasetPaths  # noqa: E402


def main() -> int:
    """Print dataset-root availability and return an appropriate exit status."""
    try:
        dataset_paths = DatasetPaths.from_environment()
    except EnvironmentError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    report = dataset_paths.existence_report()
    for dataset_name, exists in report.items():
        status = "exists" if exists else "missing"
        print(f"{dataset_name}: {status}")

    return 0 if all(report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
