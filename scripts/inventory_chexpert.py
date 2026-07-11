"""Print a filesystem-only inventory for a CheXpert dataset root."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.inventory import IMAGE_EXTENSIONS, inventory_chexpert  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the CheXpert root directory.")
    return parser.parse_args()


def main() -> int:
    """Print the inventory and return a process exit status."""
    args = parse_args()
    try:
        inventory = inventory_chexpert(args.root)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Inventory error: {error}", file=sys.stderr)
        return 1

    print(f"Root path: {inventory.root_path}")
    print(f"Total file count: {inventory.total_files}")
    print("Image count by extension:")
    for extension in IMAGE_EXTENSIONS:
        print(f"  {extension}: {inventory.image_files_by_extension[extension]}")
    print(f"CSV file count: {len(inventory.csv_files)}")
    for csv_file in inventory.csv_files:
        print(f"  {csv_file}")
    print(f"Directory count: {inventory.directory_count}")
    print("Representative paths:")
    for path in inventory.representative_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
