from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "zscore"  # zscore | minmax


class SurfaceNormalizer:
    def __init__(self, config: NormalizationConfig | None = None):
        self.config = config or NormalizationConfig()
        self._scaler: StandardScaler | MinMaxScaler | None = None

    def fit(self, surfaces: np.ndarray) -> None:
        flat = surfaces.reshape(surfaces.shape[0], -1)
        if self.config.method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            self._scaler = StandardScaler()
        self._scaler.fit(flat)

    def transform(self, surfaces: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("SurfaceNormalizer.fit must be called before transform.")
        flat = surfaces.reshape(surfaces.shape[0], -1)
        out = self._scaler.transform(flat)
        return out.reshape(surfaces.shape)

    def fit_transform(self, surfaces: np.ndarray) -> np.ndarray:
        self.fit(surfaces)
        return self.transform(surfaces)

    def inverse_transform(self, surfaces: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("SurfaceNormalizer.fit must be called before inverse_transform.")
        flat = surfaces.reshape(surfaces.shape[0], -1)
        inv = self._scaler.inverse_transform(flat)
        return inv.reshape(surfaces.shape)
