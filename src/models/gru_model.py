from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseSurfaceModel, ModelConfig


class GRUSurfaceForecaster(BaseSurfaceModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, e, k = x.shape
        seq = x.view(b, l, e * k)
        out, _ = self.gru(seq)
        return self.reshape_output(self.head(out[:, -1, :]))
