"""Model registry used by experiment runners and tests."""

from . import (
    TEFN,
    TEFN_FuzzyTCN_Classifier_32,
)

MODEL_REGISTRY = {
    "TEFN": TEFN,
    "TEFN_FuzzyTCN_Classifier_32": TEFN_FuzzyTCN_Classifier_32,
}

__all__ = [
    "MODEL_REGISTRY",
    "TEFN",
    "TEFN_FuzzyTCN_Classifier_32",
]
