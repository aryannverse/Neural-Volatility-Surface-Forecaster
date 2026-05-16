from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    rv_window: int = 20
    hv_window: int = 60


class FeatureEngineer:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def add_underlying_features(self, chain_df: pd.DataFrame, spot_history: pd.DataFrame) -> pd.DataFrame:
        out = chain_df.copy()
        px = spot_history["Close"].astype(float)
        returns = np.log(px / px.shift(1))
        rv = returns.rolling(self.config.rv_window).std() * np.sqrt(252.0)
        hv = returns.rolling(self.config.hv_window).std() * np.sqrt(252.0)
        feat = pd.DataFrame({"date": spot_history.index.date, "rv": rv.values, "hv": hv.values})
        feat = feat.dropna().groupby("date", as_index=False).last()

        out["date"] = pd.to_datetime(out["timestamp"]).dt.date
        out = out.merge(feat, on="date", how="left")
        out["log_moneyness"] = np.log(out["strike"] / out["spot"])
        out["ttm_sqrt"] = np.sqrt(out["ttm"])
        out["spread"] = (out["ask"] - out["bid"]).clip(lower=0.0)
        out["liq_score"] = (
            np.log1p(out["volume"].fillna(0.0))
            + np.log1p(out["open_interest"].fillna(0.0))
            - np.log1p(out["spread"].fillna(0.0) + 1e-6)
        )
        return out

    @staticmethod
    def surface_features(surfaces: np.ndarray) -> pd.DataFrame:
        level = surfaces.mean(axis=(1, 2))
        skew = np.median(np.gradient(surfaces, axis=2), axis=(1, 2))
        curvature = np.median(np.gradient(np.gradient(surfaces, axis=2), axis=2), axis=(1, 2))
        term_slope = np.median(np.gradient(surfaces, axis=1), axis=(1, 2))
        shock = np.abs(np.diff(level, prepend=level[0]))
        return pd.DataFrame(
            {
                "iv_level": level,
                "skew_metric": skew,
                "smile_curvature": curvature,
                "term_slope": term_slope,
                "surface_shock": shock,
            }
        )
