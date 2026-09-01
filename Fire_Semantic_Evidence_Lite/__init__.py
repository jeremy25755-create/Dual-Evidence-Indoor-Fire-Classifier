"""Lightweight semantic-evidence network for indoor-fire classification."""

from .losses import evidential_classification_loss
from .model import FireSemanticEvidenceLite

__all__ = ["FireSemanticEvidenceLite", "evidential_classification_loss"]
