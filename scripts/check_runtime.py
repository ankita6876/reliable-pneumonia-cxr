"""Report Python and PyTorch CUDA runtime information."""

import argparse
import platform

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runtime-check options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when CUDA is unavailable.",
    )
    return parser.parse_args(argv)


def check_runtime(require_cuda: bool = False) -> int:
    """Print runtime details and optionally require a CUDA-capable PyTorch install."""
    cuda_available = torch.cuda.is_available()
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    if require_cuda and not cuda_available:
        print("CUDA is required but unavailable. Enable a GPU runtime and retry.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the runtime check from the command line."""
    args = parse_args(argv)
    return check_runtime(require_cuda=args.require_cuda)


if __name__ == "__main__":
    raise SystemExit(main())
