from types import SimpleNamespace

import pytest
import torch

from models.TEFN import EvidenceMachineKernel, Model


def make_config(**overrides):
    values = {
        "task_name": "long_term_forecast",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 4,
        "enc_in": 3,
        "e_layers": 2,
        "kernel_activation": None,
        "use_residual": True,
        "use_norm": True,
        "use_T_model": True,
        "use_C_model": True,
        "fusion_method": "add",
        "use_probabilistic_layer": False,
        "dropout": 0.1,
        "num_class": 4,
        "classification_hidden_dim": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("fusion_method", ["add", "concat"])
@pytest.mark.parametrize(
    ("use_time", "use_channel"),
    [(True, True), (True, False), (False, True)],
)
def test_forward_shape_and_backward(fusion_method, use_time, use_channel):
    torch.manual_seed(7)
    model = Model(
        make_config(
            fusion_method=fusion_method,
            use_T_model=use_time,
            use_C_model=use_channel,
        )
    )
    inputs = torch.randn(2, 8, 3, requires_grad=True)

    outputs = model(inputs, None, None, None)

    assert outputs.shape == (2, 4, 3)
    assert torch.isfinite(outputs).all()
    outputs.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_both_evidence_branches_cannot_be_disabled():
    with pytest.raises(ValueError, match="At least one evidence branch"):
        Model(make_config(use_T_model=False, use_C_model=False))


def test_unknown_kernel_activation_fails_fast():
    with pytest.raises(ValueError, match="Unsupported kernel activation"):
        EvidenceMachineKernel(3, 2, activation="unknown")


def test_invalid_input_length_fails_fast():
    model = Model(make_config())
    with pytest.raises(ValueError, match="Expected seq_len"):
        model(torch.randn(2, 7, 3), None, None, None)


@pytest.mark.parametrize("fusion_method", ["add", "concat"])
def test_classification_shape_and_backward(fusion_method):
    torch.manual_seed(11)
    model = Model(
        make_config(
            task_name="classification",
            fusion_method=fusion_method,
            seq_len=8,
            enc_in=3,
        )
    )
    inputs = torch.randn(5, 8, 3, requires_grad=True)

    logits = model(inputs, None, None, None)

    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
