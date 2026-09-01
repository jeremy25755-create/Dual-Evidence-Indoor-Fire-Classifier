"""Losses for Dirichlet semantic evidence classification."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)) for each sample."""
    classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1, keepdim=True)
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        - torch.lgamma(torch.tensor(float(classes), device=alpha.device))
    )
    divergence = log_normalizer + (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
    ).sum(dim=-1, keepdim=True)
    return divergence.squeeze(-1)


def evidential_classification_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    annealing: float = 1.0,
    kl_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expected cross entropy plus annealed incorrect-evidence regularization."""
    classes = alpha.shape[-1]
    one_hot = functional.one_hot(targets, num_classes=classes).to(alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    expected_ce = (
        one_hot * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    if class_weights is not None:
        expected_ce = expected_ce * class_weights[targets]

    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(adjusted_alpha)
    classification_term = expected_ce.mean()
    regularization_term = kl.mean()
    loss = classification_term + float(annealing) * kl_weight * regularization_term
    return loss, {
        "classification": classification_term.detach(),
        "kl": regularization_term.detach(),
    }
