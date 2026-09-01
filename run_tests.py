"""Run pytest after loading PyTorch first.

PyTorch 2.11 CUDA wheels can fail to initialize c10.dll on some Windows
machines when another native-extension stack is imported before torch.
"""

import sys

import torch

_ = torch.__version__

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
