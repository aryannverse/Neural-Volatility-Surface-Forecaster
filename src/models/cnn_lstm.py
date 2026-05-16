from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseSurfaceModel, ModelConfig


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNLSTMForecaster(BaseSurfaceModel):
    """
    Spatial-temporal architecture:
    - CNN extracts local skew/smile structures
    - LSTM models temporal deformation of those structures
    """

    def __init__(self, cfg: ModelConfig, cnn_latent: int = 128):
        super().__init__(cfg)
        self.encoder = CNNEncoder(latent_dim=cnn_latent)
        self.lstm = nn.LSTM(
            input_size=cnn_latent,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, e, k = x.shape
        frames = x.reshape(b * l, 1, e, k)
        spatial = self.encoder(frames).reshape(b, l, -1)
        temporal, _ = self.lstm(spatial)
        y = self.head(temporal[:, -1, :])
        return self.reshape_output(y)
