from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.iv_surface.interpolation import SurfaceInterpolator


@dataclass(frozen=True)
class SurfaceGridSpec:
    strike_points: int = 41
    expiry_points: int = 24
    strike_min_moneyness: float = 0.7
    strike_max_moneyness: float = 1.3
    expiry_min_years: float = 1.0 / 365.25
    expiry_max_years: float = 2.0
    use_moneyness: bool = True


@dataclass(frozen=True)
class SurfaceSnapshot:
    ticker: str
    timestamp: datetime
    strike_axis: np.ndarray
    expiry_axis: np.ndarray
    iv_grid: np.ndarray  # [expiry, strike]


class SurfaceBuilder:
    def __init__(self, grid_spec: SurfaceGridSpec | None = None, interpolator: SurfaceInterpolator | None = None):
        self.grid_spec = grid_spec or SurfaceGridSpec()
        self.interpolator = interpolator or SurfaceInterpolator()

    def _grid_axes(self, chain_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        expiry_axis = np.linspace(
            self.grid_spec.expiry_min_years,
            self.grid_spec.expiry_max_years,
            self.grid_spec.expiry_points,
        )
        if self.grid_spec.use_moneyness:
            strike_axis = np.linspace(
                self.grid_spec.strike_min_moneyness,
                self.grid_spec.strike_max_moneyness,
                self.grid_spec.strike_points,
            )
        else:
            strike_axis = np.linspace(
                max(1e-6, float(chain_df["strike"].quantile(0.02))),
                float(chain_df["strike"].quantile(0.98)),
                self.grid_spec.strike_points,
            )
        return strike_axis, expiry_axis

    def build_surface(self, chain_df: pd.DataFrame) -> SurfaceSnapshot:
        required = {"ticker", "timestamp", "iv", "ttm", "strike", "moneyness"}
        missing_cols = required - set(chain_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns for surface construction: {missing_cols}")

        frame = chain_df.copy()
        frame = frame[np.isfinite(frame["iv"]) & (frame["iv"] > 0.0)]
        frame = frame[np.isfinite(frame["ttm"]) & (frame["ttm"] > 0.0)]

        strike_axis, expiry_axis = self._grid_axes(frame)
        x_col = "moneyness" if self.grid_spec.use_moneyness else "strike"

        pivot = (
            frame.assign(
                strike_bin=pd.cut(frame[x_col], bins=strike_axis, include_lowest=True),
                expiry_bin=pd.cut(frame["ttm"], bins=expiry_axis, include_lowest=True),
            )
            .groupby(["expiry_bin", "strike_bin"], observed=False)["iv"]
            .median()
            .unstack()
        )

        z_grid = np.full((len(expiry_axis), len(strike_axis)), np.nan, dtype=float)
        for i in range(min(len(pivot.index), z_grid.shape[0])):
            row = pivot.iloc[i].to_numpy(dtype=float)
            z_grid[i, : min(len(row), z_grid.shape[1])] = row[: z_grid.shape[1]]

        z_interp = self.interpolator.interpolate_surface(
            x_axis=strike_axis,
            y_axis=expiry_axis,
            z_grid=z_grid,
        )

        return SurfaceSnapshot(
            ticker=str(frame["ticker"].iloc[0]),
            timestamp=pd.Timestamp(frame["timestamp"].iloc[0]).to_pydatetime(),
            strike_axis=strike_axis,
            expiry_axis=expiry_axis,
            iv_grid=z_interp,
        )

    @staticmethod
    def sequence_to_tensor(snapshots: list[SurfaceSnapshot]) -> np.ndarray:
        snapshots_sorted = sorted(snapshots, key=lambda s: s.timestamp)
        return np.stack([s.iv_grid for s in snapshots_sorted], axis=0)

    @staticmethod
    def save_snapshot(snapshot: SurfaceSnapshot, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{snapshot.ticker}_{snapshot.timestamp:%Y%m%d_%H%M%S}.npz"
        np.savez_compressed(
            out,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp.isoformat(),
            strike_axis=snapshot.strike_axis,
            expiry_axis=snapshot.expiry_axis,
            iv_grid=snapshot.iv_grid,
        )
        return out

    @staticmethod
    def load_snapshot(path: Path) -> SurfaceSnapshot:
        payload = np.load(path, allow_pickle=True)
        return SurfaceSnapshot(
            ticker=str(payload["ticker"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            strike_axis=payload["strike_axis"].astype(float),
            expiry_axis=payload["expiry_axis"].astype(float),
            iv_grid=payload["iv_grid"].astype(float),
        )
