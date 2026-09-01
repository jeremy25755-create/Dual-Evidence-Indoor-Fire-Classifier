import pytest
import torch

from Fire_Semantic_Evidence_Lite.losses import evidential_classification_loss
from Fire_Semantic_Evidence_Lite.model import FireSemanticEvidenceLite


def test_semantic_evidence_shape_probability_and_backward():
    model = FireSemanticEvidenceLite()
    inputs = torch.randn(5, 60, 14, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1, 0])

    output = model(inputs)
    loss, terms = evidential_classification_loss(output["alpha"], targets)

    assert output["evidence"].shape == (5, 3)
    assert output["probabilities"].shape == (5, 3)
    assert output["uncertainty"].shape == (5, 1)
    assert output["level_evidence"].shape == (5, 3)
    assert output["trend_evidence"].shape == (5, 3)
    assert torch.allclose(
        output["level_weight"] + output["trend_weight"],
        torch.ones(5, 1),
        atol=1e-6,
    )
    assert torch.allclose(
        output["probabilities"].sum(dim=-1), torch.ones(5), atol=1e-6
    )
    assert torch.all(output["evidence"] >= 0)
    assert torch.all((output["uncertainty"] > 0) & (output["uncertainty"] <= 1))
    assert torch.isfinite(terms["classification"])
    assert torch.isfinite(terms["kl"])
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_semantic_evidence_model_is_lightweight():
    model = FireSemanticEvidenceLite()
    parameters = sum(parameter.numel() for parameter in model.parameters())

    assert 11_000 <= parameters <= 13_000


def test_semantic_evidence_rejects_wrong_shape():
    model = FireSemanticEvidenceLite()

    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(2, 59, 14))
