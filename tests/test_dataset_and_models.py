from __future__ import annotations

import numpy as np
import torch

from src.datasets.surface_dataset import DatasetConfig, create_dataloaders
from src.models.base import ModelConfig
from src.models.factory import build_model


def test_dataloader_and_model_shapes() -> None:
    surfaces = np.random.rand(120, 24, 41).astype(np.float32)
    train_loader, _, _ = create_dataloaders(
        surfaces, DatasetConfig(lookback=20, horizon=1, batch_size=8)
    )
    x, y = next(iter(train_loader))
    assert x.shape == (8, 20, 24, 41)
    assert y.shape == (8, 24, 41)

    cfg = ModelConfig(lookback=20, expiry_dim=24, strike_dim=41)
    for name in ["lstm", "gru", "cnn_lstm", "transformer", "autoencoder"]:
        model = build_model(name, cfg)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 24, 41)
