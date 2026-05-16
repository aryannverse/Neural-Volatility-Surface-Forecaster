from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetConfig:
    lookback: int = 20
    horizon: int = 1
    batch_size: int = 32
    num_workers: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    shuffle_train: bool = True


class SurfaceForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, surfaces: np.ndarray, lookback: int, horizon: int = 1):
        if surfaces.ndim != 3:
            raise ValueError("surfaces must be [time, expiry, strike]")
        self.surfaces = surfaces.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self._length = len(surfaces) - lookback - horizon + 1
        if self._length <= 0:
            raise ValueError("Not enough surface points for requested lookback and horizon.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.surfaces[idx : idx + self.lookback]  # [L, E, K]
        y = self.surfaces[idx + self.lookback + self.horizon - 1]  # [E, K]
        return torch.from_numpy(x), torch.from_numpy(y)


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def create_dataloaders(surfaces: np.ndarray, cfg: DatasetConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    ds = SurfaceForecastDataset(surfaces=surfaces, lookback=cfg.lookback, horizon=cfg.horizon)
    n = len(ds)
    s_train, s_val, s_test = _split_indices(n, cfg.train_ratio, cfg.val_ratio)
    idx = np.arange(n)

    train_subset = torch.utils.data.Subset(ds, idx[s_train])
    val_subset = torch.utils.data.Subset(ds, idx[s_val])
    test_subset = torch.utils.data.Subset(ds, idx[s_test])

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def rolling_window_splits(
    dataset_len: int,
    train_size: int,
    val_size: int,
    test_size: int,
    step: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    while start + train_size + val_size + test_size <= dataset_len:
        train_idx = np.arange(start, start + train_size)
        val_idx = np.arange(start + train_size, start + train_size + val_size)
        test_idx = np.arange(start + train_size + val_size, start + train_size + val_size + test_size)
        splits.append((train_idx, val_idx, test_idx))
        start += step
    return splits
