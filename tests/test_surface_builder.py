from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.iv_surface.builder import SurfaceBuilder


def test_surface_generation_shape() -> None:
    n = 300
    df = pd.DataFrame(
        {
            "ticker": ["SPY"] * n,
            "timestamp": [datetime.now(timezone.utc)] * n,
            "strike": np.random.uniform(80, 120, n),
            "spot": [100.0] * n,
            "moneyness": np.random.uniform(0.8, 1.2, n),
            "ttm": np.random.uniform(0.02, 1.5, n),
            "iv": np.random.uniform(0.1, 0.4, n),
        }
    )
    snap = SurfaceBuilder().build_surface(df)
    assert snap.iv_grid.ndim == 2
    assert snap.iv_grid.shape[0] == len(snap.expiry_axis)
    assert snap.iv_grid.shape[1] == len(snap.strike_axis)
    assert np.isfinite(snap.iv_grid).all()
