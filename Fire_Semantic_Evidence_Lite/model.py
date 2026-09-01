"""Level-trend semantic evidence classifier without TEFN BPA modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class LevelEncoder(nn.Module):
    """Encode absolute sensor levels with weights shared over time."""

    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(channels, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        features = self.encoder(window)
        return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)


class TrendEncoder(nn.Module):
    """Encode multi-scale first-difference patterns with depthwise convolutions."""

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kernels: tuple[int, ...] = (3, 5, 9),
    ):
        super().__init__()
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Trend kernels must be positive odd integers")
        self.depthwise = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=channels,
            )
            for kernel in kernels
        )
        self.pointwise = nn.Conv1d(
            channels * len(kernels), embedding_dim, kernel_size=1
        )
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        differences = torch.diff(window, dim=1).transpose(1, 2)
        multi_scale = torch.cat(
            [convolution(differences) for convolution in self.depthwise], dim=1
        )
        features = self.pointwise(multi_scale).transpose(1, 2)
        features = self.normalization(self.activation(features))
        return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)


class SemanticEvidenceHead(nn.Module):
    """Produce non-negative class evidence and derived Dirichlet uncertainty."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.evidence_activation = nn.Softplus()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.evidence_activation(self.network(features))
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probabilities = alpha / strength
        uncertainty = self.num_classes / strength
        return {
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
        }


class FireSemanticEvidenceLite(nn.Module):
    """Fuse independent level and trend semantic evidence by confidence."""

    def __init__(
        self,
        seq_len: int = 60,
        channels: int = 14,
        embedding_dim: int = 32,
        evidence_hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for trend encoding")
        if min(channels, embedding_dim, evidence_hidden_dim, num_classes) < 1:
            raise ValueError("All model dimensions must be positive")
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.level_encoder = LevelEncoder(channels, embedding_dim)
        self.trend_encoder = TrendEncoder(channels, embedding_dim)
        pooled_dim = 2 * embedding_dim
        self.level_evidence_head = SemanticEvidenceHead(
            input_dim=pooled_dim,
            hidden_dim=evidence_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.trend_evidence_head = SemanticEvidenceHead(
            input_dim=pooled_dim,
            hidden_dim=evidence_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _fuse_evidence(
        self,
        level_output: dict[str, torch.Tensor],
        trend_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        level_confidence = 1.0 - level_output["uncertainty"]
        trend_confidence = 1.0 - trend_output["uncertainty"]
        confidence_sum = level_confidence + trend_confidence + 1e-8
        level_weight = level_confidence / confidence_sum
        trend_weight = trend_confidence / confidence_sum
        evidence = (
            level_weight * level_output["evidence"]
            + trend_weight * trend_output["evidence"]
        )
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return {
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": alpha / strength,
            "uncertainty": self.num_classes / strength,
            "level_weight": level_weight,
            "trend_weight": trend_weight,
        }

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        if window.ndim != 3 or tuple(window.shape[1:]) != (
            self.seq_len,
            self.channels,
        ):
            raise ValueError(
                f"Expected input [batch, {self.seq_len}, {self.channels}], "
                f"received {tuple(window.shape)}"
            )
        level_features = self.level_encoder(window)
        trend_features = self.trend_encoder(window)
        level_output = self.level_evidence_head(level_features)
        trend_output = self.trend_evidence_head(trend_features)
        output = self._fuse_evidence(level_output, trend_output)
        output["level_features"] = level_features
        output["trend_features"] = trend_features
        output["level_evidence"] = level_output["evidence"]
        output["trend_evidence"] = trend_output["evidence"]
        output["level_alpha"] = level_output["alpha"]
        output["trend_alpha"] = trend_output["alpha"]
        output["level_uncertainty"] = level_output["uncertainty"]
        output["trend_uncertainty"] = trend_output["uncertainty"]
        return output
