from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseSurfaceModel, ModelConfig


class Conv3DSurfaceForecaster(BaseSurfaceModel):
    """
    Advanced spatio-temporal alternative using 3D convolutions over [time, expiry, strike].
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, E, K] -> [B, 1, L, E, K]
        y = self.net(x.unsqueeze(1))
        return self.reshape_output(y)
