from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseSurfaceModel, ModelConfig


class TransformerSurfaceForecaster(BaseSurfaceModel):
    """
    Attention-based forecaster. Self-attention learns non-local temporal dependencies
    and regime-dependent interactions in surface evolution.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        ff_dim: int = 512,
    ):
        super().__init__(cfg)
        self.proj = nn.Linear(cfg.input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.lookback, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, e, k = x.shape
        seq = x.view(b, l, e * k)
        h = self.proj(seq) + self.pos_emb[:, :l, :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        y = self.head(h)
        return self.reshape_output(y)
