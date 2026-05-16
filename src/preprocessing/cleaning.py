from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class CleaningConfig:
    iv_min: float = 1e-4
    iv_max: float = 5.0
    zscore_clip: float = 4.5
    surface_median_kernel: int = 3


class SurfaceCleaner:
    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    def clean_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        numeric_cols = ["strike", "bid", "ask", "mid_price", "iv", "ttm", "volume", "open_interest"]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["strike", "ttm", "iv"])
        out = out[(out["iv"] >= self.config.iv_min) & (out["iv"] <= self.config.iv_max)]
        out = out[np.isfinite(out["iv"])]

        # Robust outlier clipping by expiry bucket to preserve smile shape.
        grouped = out.groupby(pd.cut(out["ttm"], bins=10), observed=False)["iv"]
        bounds = grouped.transform(
            lambda x: x.median()
            + self.config.zscore_clip * 1.4826 * np.median(np.abs(x - x.median()))
        )
        out = out[out["iv"] <= bounds]
        return out.reset_index(drop=True)

    def smooth_surface(self, iv_grid: np.ndarray) -> np.ndarray:
        smoothed = median_filter(iv_grid, size=self.config.surface_median_kernel)
        return np.clip(smoothed, self.config.iv_min, self.config.iv_max)
