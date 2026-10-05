"""Compact self-contained fuzzy dual-evidence TCN classifier."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mix(nn.Module):
    def __init__(self, steps, states):
        super().__init__()
        self.time, self.state = nn.Linear(steps, steps), nn.Linear(states, states)

    def forward(self, x):
        return self.state(self.time(x.transpose(-2, -1)).transpose(-2, -1))


class BaseTime(nn.Module):
    def __init__(self, steps, states):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(steps, states))
        self.bias = nn.Parameter(torch.randn(steps, states))
        self.mix = Mix(steps, states)

    def forward(self, x):
        x = torch.einsum("btc,tf->bctf", x, self.weight) + self.bias
        return (self.mix(x) + x).permute(0, 2, 1, 3)


class Causal(nn.Module):
    def __init__(self, dim, dilation, dropout):
        super().__init__()
        self.depth = nn.Conv1d(dim, dim, 3, padding=2 * dilation,
                               dilation=dilation, groups=dim)
        self.point = nn.Conv1d(dim, dim, 1)
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, x):
        y = self.depth(x)[..., :x.size(-1)]
        y = self.norm(self.point(y).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(y))


class TimeTCN(nn.Module):
    def __init__(self, states, hidden, dropout):
        super().__init__()
        if hidden < 1:
            raise ValueError("tcn_hidden_dim must be positive")
        self.receptive_field = 31
        self.stem = nn.Conv1d(1, hidden, 1)
        self.blocks = nn.Sequential(
            *(Causal(hidden, d, dropout) for d in (1, 2, 4, 8))
        )
        self.out = nn.Conv1d(hidden, states, 1)

    def forward(self, x):
        batch, steps, sensors = x.shape
        x = x.permute(0, 2, 1).reshape(batch * sensors, 1, steps)
        x = self.out(self.blocks(self.stem(x)))
        return x.reshape(batch, sensors, -1, steps).permute(0, 3, 1, 2)


class HybridTime(nn.Module):
    def __init__(self, steps, states, hidden, dropout):
        super().__init__()
        self.base = BaseTime(steps, states)
        self.delta = TimeTCN(states, hidden, dropout)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        change = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 1, 0))
        return self.base(x) + torch.sigmoid(self.gate_logit) * self.delta(change)


class Fuzzy(nn.Module):
    def __init__(self, sensors, levels):
        super().__init__()
        if levels < 2:
            raise ValueError("fuzzy_levels must be at least 2")
        self.register_buffer(
            "base_centers", torch.linspace(-1.0, 1.0, levels).repeat(sensors, 1)
        )
        self.center_offsets = nn.Parameter(torch.zeros(sensors, levels))
        self.raw_widths = nn.Parameter(torch.full((sensors, levels), -0.5))

    def centers(self):
        return self.base_centers + 0.25 * torch.tanh(self.center_offsets)

    def widths(self):
        return 0.1 + F.softplus(self.raw_widths)

    def forward(self, x):
        distance = (x[..., None] - self.centers()[None, None]) \
                   / self.widths()[None, None]
        return torch.softmax(-0.5 * distance.square(), dim=-1)


class Sensor(nn.Module):
    def __init__(self, sensors, levels, states):
        super().__init__()
        self.levels, self.fuzzy = levels, Fuzzy(sensors, levels)
        self.fuzzy_weight = nn.Parameter(torch.randn(sensors, levels, states))
        self.fuzzy_bias = nn.Parameter(torch.randn(sensors, states))
        self.cont_weight = nn.Parameter(torch.randn(sensors, states))
        self.cont_bias = nn.Parameter(torch.randn(sensors, states))

    def forward(self, raw, normalized):
        membership = self.fuzzy(raw)
        fuzzy = torch.einsum(
            "btcl,clf->btcf", membership, self.fuzzy_weight
        ) + self.fuzzy_bias
        continuous = torch.einsum(
            "btc,cf->btcf", torch.sigmoid(normalized), self.cont_weight
        ) + self.cont_bias
        entropy = -(membership * membership.clamp_min(1e-8).log()).sum(-1, True)
        confidence = (1.0 - entropy / math.log(self.levels)).clamp(0.0, 1.0)
        return confidence * fuzzy + (1.0 - confidence) * continuous


class Fuse(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.linear = nn.Linear(2 * states, states)

    def forward(self, time, sensor):
        return self.linear(torch.cat((time, sensor), dim=-1))


class Head(nn.Module):
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, x):
        logits = self.net(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        strength = alpha.sum(-1, keepdim=True)
        return {"logits": logits, "signed_evidence": logits,
                "support": F.relu(logits), "opposition": F.relu(-logits),
                "evidence": evidence, "alpha": alpha,
                "probabilities": alpha / strength, "uncertainty": 3.0 / strength}


class Net(nn.Module):
    def __init__(self, sensors=14, power=2, levels=3, hidden=64,
                 tcn_hidden=8, dropout=0.1):
        super().__init__()
        self.steps, self.sensors, self.states = 32, sensors, 2**power
        self.time = HybridTime(32, self.states, tcn_hidden, dropout)
        self.sensor = Sensor(sensors, levels, self.states)
        self.fuse = Fuse(self.states)
        features = 2 * sensors * self.states + 2 * sensors
        self.head = Head(features, hidden, dropout)

    @staticmethod
    def normalize(x):
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / std

    def forward(self, x, *_args, **_kwargs):
        if x.ndim != 3 or x.shape[1:] != (32, self.sensors):
            raise ValueError(f"Expected [B, 32, {self.sensors}], got {tuple(x.shape)}")
        normalized = self.normalize(x)
        fused = self.fuse(self.time(normalized), self.sensor(x, normalized))
        features = torch.cat((fused.mean(1).flatten(1), fused.amax(1).flatten(1),
                              x.mean(1), torch.sqrt(x.var(1, unbiased=False) + 1e-5)), 1)
        return self.head(features)

    def fuzzy_parameters(self):
        return {"centers": self.sensor.fuzzy.centers().detach().cpu(),
                "widths": self.sensor.fuzzy.widths().detach().cpu()}


def Model(configs):
    if configs.seq_len != 32 or configs.num_class != 3:
        raise ValueError("This model requires --seq_len 32 --num_class 3")
    return Net(configs.enc_in, configs.e_layers, getattr(configs, "fuzzy_levels", 3),
               configs.classification_hidden_dim, getattr(configs, "tcn_hidden_dim", 8),
               configs.dropout)
