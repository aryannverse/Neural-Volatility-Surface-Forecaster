from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SurfaceTensorBundle:
    ticker: str
    strike_axis: np.ndarray
    expiry_axis: np.ndarray
    timestamps: np.ndarray
    tensor: np.ndarray  # [time, expiry, strike]


class SurfaceTensorStore:
    @staticmethod
    def save(bundle: SurfaceTensorBundle, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ticker=bundle.ticker,
            strike_axis=bundle.strike_axis,
            expiry_axis=bundle.expiry_axis,
            timestamps=bundle.timestamps.astype("datetime64[s]").astype(str),
            tensor=bundle.tensor,
        )

    @staticmethod
    def load(path: Path) -> SurfaceTensorBundle:
        data = np.load(path, allow_pickle=True)
        return SurfaceTensorBundle(
            ticker=str(data["ticker"]),
            strike_axis=data["strike_axis"],
            expiry_axis=data["expiry_axis"],
            timestamps=data["timestamps"],
            tensor=data["tensor"],
        )
