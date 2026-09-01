"""Time Evidence Fusion Network baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class NormLayer(nn.Module):
    """Per-sample temporal normalization with explicit statistics."""

    @staticmethod
    def norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stds = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return centered / stds, means, stds

    @staticmethod
    def denorm(x: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
        return x * stds + means


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("MLP requires at least one hidden layer")

        modules: list[nn.Module] = []
        for index in range(layers):
            modules.append(
                nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim)
            )
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(scores)
        return self.linear(torch.matmul(attention_weights, value))


class EvidenceMachineKernel(nn.Module):
    """Expand each source into evidence states and apply an optional transform."""

    def __init__(
        self,
        sources: int,
        evidence_power: int,
        activation: str | None = None,
        use_residual: bool = True,
    ):
        super().__init__()
        if sources < 1:
            raise ValueError("sources must be positive")
        if evidence_power < 0:
            raise ValueError("evidence_power must be non-negative")

        # Keep the original names for compatibility with official TEFN checkpoints.
        self.C = sources
        self.F = 2**evidence_power
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))
        self.use_residual = use_residual
        self.activation = self._make_activation(activation)

    def _make_activation(self, activation: str | None) -> nn.Module | None:
        merged_dim = self.C * self.F
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": Swish(),
            "mish": Mish(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "linear": nn.Linear(merged_dim, merged_dim),
            "mlp": MLP(merged_dim, 2 * merged_dim, merged_dim, 2),
            "attn": Attention(merged_dim),
        }
        if activation is None:
            return None
        if activation not in activations:
            raise ValueError(
                f"Unsupported kernel activation {activation!r}. "
                f"Choose from: {', '.join(activations)}"
            )
        return activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.C:
            raise ValueError(
                f"Expected {self.C} sources in the last dimension, "
                f"received {x.shape[-1]}"
            )

        evidence = (
            torch.einsum("btc,cf->btcf", x, self.C_weight)
            + self.C_bias
        )
        batch, steps, sources, states = evidence.shape
        merged = evidence.reshape(batch, steps, -1)
        if self.activation is not None:
            activated = self.activation(merged)
            merged = activated + merged if self.use_residual else activated
        return merged.reshape(batch, steps, sources, states)


class Model(nn.Module):
    """TEFN model for forecasting or multivariate time-series classification."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.use_norm = bool(configs.use_norm)
        self.use_T_model = bool(configs.use_T_model)
        self.use_C_model = bool(configs.use_C_model)
        self.fusion_method = configs.fusion_method
        self.use_probabilistic_layer = bool(configs.use_probabilistic_layer)
        self.is_classification = self.task_name == "classification"

        if not (
            self.task_name.startswith("long_term_forecast")
            or self.is_classification
        ):
            raise ValueError(f"TEFN does not support task_name={self.task_name!r}")
        if not (self.use_T_model or self.use_C_model):
            raise ValueError("At least one evidence branch must be enabled")
        if self.fusion_method not in {"add", "concat"}:
            raise ValueError("fusion_method must be 'add' or 'concat'")

        if self.use_norm:
            self.norm_layer = NormLayer()
        evidence_steps = self.seq_len if self.is_classification else self.pred_len + self.seq_len
        if not self.is_classification:
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len
            )
        self.T_model = EvidenceMachineKernel(
            evidence_steps,
            configs.e_layers,
            activation=configs.kernel_activation,
            use_residual=configs.use_residual,
        )
        self.C_model = EvidenceMachineKernel(
            configs.enc_in,
            configs.e_layers,
            activation=configs.kernel_activation,
            use_residual=configs.use_residual,
        )
        if self.fusion_method == "concat":
            evidence_states = 2**configs.e_layers
            self.fusion_linear = nn.Linear(2 * evidence_states, evidence_states)
        if self.use_probabilistic_layer:
            self.probabilistic_layer = nn.Dropout(p=configs.dropout)
        if self.is_classification:
            evidence_states = 2**configs.e_layers
            pooled_dim = 2 * configs.enc_in * evidence_states + 2 * configs.enc_in
            hidden_dim = int(configs.classification_hidden_dim)
            self.classification_head = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=configs.dropout),
                nn.Linear(hidden_dim, int(configs.num_class)),
            )

    def _fuse_evidence(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs: list[torch.Tensor] = []
        if self.use_T_model:
            time_evidence = self.T_model(x.transpose(1, 2)).permute(0, 2, 1, 3)
            branch_outputs.append(time_evidence)
        if self.use_C_model:
            branch_outputs.append(self.C_model(x))

        if len(branch_outputs) == 1:
            fused = branch_outputs[0]
        elif self.fusion_method == "add":
            fused = branch_outputs[0] + branch_outputs[1]
        else:
            fused = self.fusion_linear(torch.cat(branch_outputs, dim=-1))
        if self.use_probabilistic_layer:
            fused = self.probabilistic_layer(fused)
        return fused

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3:
            raise ValueError(
                f"x_enc must have shape [batch, time, channel], got {tuple(x_enc.shape)}"
            )
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, received {x_enc.shape[1]}"
            )

        if self.use_norm:
            x, means, stds = self.norm_layer.norm(x_enc)
        else:
            x = x_enc
            means = stds = None

        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)
        fused = self._fuse_evidence(x)

        output = fused.sum(dim=-1)
        if self.use_norm:
            output = self.norm_layer.denorm(output, means, stds)
        return output

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError(
                f"x_enc must have shape [batch, time, channel], got {tuple(x_enc.shape)}"
            )
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.configs.enc_in:
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.configs.enc_in}], "
                f"received {tuple(x_enc.shape)}"
            )
        raw_mean = x_enc.mean(dim=1)
        raw_std = torch.sqrt(torch.var(x_enc, dim=1, unbiased=False) + 1e-5)
        if self.use_norm:
            x, _, _ = self.norm_layer.norm(x_enc)
        else:
            x = x_enc
        fused = self._fuse_evidence(x)
        pooled = torch.cat(
            (
                fused.mean(dim=1).flatten(start_dim=1),
                fused.amax(dim=1).flatten(start_dim=1),
                raw_mean,
                raw_std,
            ),
            dim=-1,
        )
        return self.classification_head(pooled)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        if self.is_classification:
            return self.classification(x_enc)
        output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return output[:, -self.pred_len :, :]
