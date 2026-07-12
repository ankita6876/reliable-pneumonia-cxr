"""Tests for the CUDA runtime preflight check."""

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_runtime  # noqa: E402


def test_runtime_check_reports_cpu_runtime_without_cuda(monkeypatch, capsys) -> None:
    """CPU-only environments report details and succeed unless CUDA is required."""
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    monkeypatch.setattr(check_runtime, "torch", fake_torch)

    result = check_runtime.check_runtime()

    output = capsys.readouterr().out
    assert result == 0
    assert "Python version:" in output
    assert "PyTorch version: test-torch" in output
    assert "CUDA available: False" in output
    assert "CUDA device count: 0" in output
    assert "CUDA device name:" not in output


def test_runtime_check_requires_cuda_when_requested(monkeypatch, capsys) -> None:
    """The optional flag fails clearly when the mocked runtime has no GPU."""
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    monkeypatch.setattr(check_runtime, "torch", fake_torch)

    result = check_runtime.check_runtime(require_cuda=True)

    assert result == 1
    assert "CUDA is required but unavailable" in capsys.readouterr().out


def test_runtime_check_reports_available_cuda_device(monkeypatch, capsys) -> None:
    """A mocked GPU exposes its device count and name without physical hardware."""
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "Test GPU",
        ),
    )
    monkeypatch.setattr(check_runtime, "torch", fake_torch)

    result = check_runtime.check_runtime(require_cuda=True)

    output = capsys.readouterr().out
    assert result == 0
    assert "CUDA available: True" in output
    assert "CUDA device count: 1" in output
    assert "CUDA device name: Test GPU" in output
