"""Small Torch model for M1c calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CalibrationNetConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 16
    output_dim: int = 2
    dropout: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationNetConfig":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=tuple(int(value) for value in payload.get("hidden_dims", (64, 32))),
            latent_dim=int(payload.get("latent_dim", 16)),
            output_dim=int(payload.get("output_dim", 2)),
            dropout=float(payload.get("dropout", 0.10)),
        )


class CalibrationNet(nn.Module):
    """A compact MLP with a latent layer suitable for future probe decoding."""

    def __init__(self, config: CalibrationNetConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.latent = nn.Linear(current_dim, config.latent_dim)
        self.latent_activation = nn.GELU()
        self.classifier = nn.Linear(config.latent_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        latent = self.latent_activation(self.latent(hidden))
        logits = self.classifier(latent)
        return logits, latent

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def output_dim(self) -> int:
        return self.config.output_dim
