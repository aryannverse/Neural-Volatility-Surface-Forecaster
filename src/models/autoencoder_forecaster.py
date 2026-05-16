from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseSurfaceModel, ModelConfig


class SurfaceEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SurfaceDecoder(nn.Module):
    def __init__(self, latent_dim: int, expiry_dim: int, strike_dim: int):
        super().__init__()
        self.expiry_dim = expiry_dim
        self.strike_dim = strike_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, expiry_dim * strike_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(out.shape[0], self.expiry_dim, self.strike_dim)


class AutoencoderForecaster(BaseSurfaceModel):
    """
    Latent-dynamics approach:
    - Encode each surface into latent manifold
    - Forecast next latent state
    - Decode latent state back into full surface
    """

    def __init__(self, cfg: ModelConfig, latent_dim: int = 96):
        super().__init__(cfg)
        self.encoder = SurfaceEncoder(latent_dim=latent_dim)
        self.temporal = nn.GRU(
            input_size=latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.latent_head = nn.Linear(cfg.hidden_dim, latent_dim)
        self.decoder = SurfaceDecoder(
            latent_dim=latent_dim,
            expiry_dim=cfg.expiry_dim,
            strike_dim=cfg.strike_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, e, k = x.shape
        frame = x.reshape(b * l, 1, e, k)
        z = self.encoder(frame).reshape(b, l, -1)
        h, _ = self.temporal(z)
        next_latent = self.latent_head(h[:, -1, :])
        return self.decoder(next_latent)
