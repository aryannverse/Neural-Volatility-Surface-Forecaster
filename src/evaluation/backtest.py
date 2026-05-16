from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.surface_dataset import SurfaceForecastDataset
from src.evaluation.metrics import regression_metrics


@dataclass(frozen=True)
class WalkForwardConfig:
    lookback: int = 20
    horizon: int = 1
    train_size: int = 120
    val_size: int = 30
    test_size: int = 30
    step: int = 20
    batch_size: int = 32


def walk_forward_windows(total_len: int, cfg: WalkForwardConfig) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    while start + cfg.train_size + cfg.val_size + cfg.test_size <= total_len:
        train = np.arange(start, start + cfg.train_size)
        val = np.arange(start + cfg.train_size, start + cfg.train_size + cfg.val_size)
        test = np.arange(
            start + cfg.train_size + cfg.val_size,
            start + cfg.train_size + cfg.val_size + cfg.test_size,
        )
        windows.append((train, val, test))
        start += cfg.step
    return windows


@torch.no_grad()
def evaluate_model_on_subset(
    model: torch.nn.Module,
    dataset: SurfaceForecastDataset,
    indices: np.ndarray,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    pred_all: list[np.ndarray] = []
    true_all: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        pred_all.append(pred)
        true_all.append(y.numpy())
    return regression_metrics(np.concatenate(true_all), np.concatenate(pred_all))
