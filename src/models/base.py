from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    lookback: int
    expiry_dim: int
    strike_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    @property
    def input_dim(self) -> int:
        return self.expiry_dim * self.strike_dim


class BaseSurfaceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def output_shape(self) -> tuple[int, int]:
        return self.cfg.expiry_dim, self.cfg.strike_dim

    def reshape_output(self, y: torch.Tensor) -> torch.Tensor:
        return y.view(y.shape[0], self.cfg.expiry_dim, self.cfg.strike_dim)
