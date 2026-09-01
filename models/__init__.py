"""Model registry used by experiment runners and tests."""

from . import TEFN

MODEL_REGISTRY = {
    "TEFN": TEFN,
}

__all__ = ["MODEL_REGISTRY", "TEFN"]
