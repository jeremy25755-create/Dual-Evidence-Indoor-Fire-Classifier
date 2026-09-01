"""Validate the project-specific TEFN CUDA environment."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    expected_venv = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix or active_prefix != expected_venv:
        raise RuntimeError(
            f"wrong virtual environment: {active_prefix}; expected {expected_venv}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the TEFN virtual environment")

    device = torch.device("cuda")
    sample = torch.randn(1024, 1024, device=device)
    result = sample @ sample
    torch.cuda.synchronize()
    print(f"python: {sys.version.split()[0]}")
    print(f"venv: {active_prefix}")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA smoke result: {result.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
