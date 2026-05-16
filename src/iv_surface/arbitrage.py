from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ArbitrageReport:
    calendar_violations: int
    butterfly_violations: int
    total_points: int

    @property
    def violation_ratio(self) -> float:
        if self.total_points == 0:
            return 0.0
        return float((self.calendar_violations + self.butterfly_violations) / self.total_points)


def check_static_arbitrage(iv_grid: np.ndarray, expiry_axis: np.ndarray) -> ArbitrageReport:
    """
    Static no-arbitrage diagnostics:
    - Calendar spread: total variance should be non-decreasing in maturity
    - Butterfly: convexity proxy by second derivative in strike dimension
    """
    total_variance = (iv_grid**2) * expiry_axis[:, None]
    calendar = np.diff(total_variance, axis=0)
    calendar_violations = int((calendar < -1e-6).sum())

    d2 = np.diff(iv_grid, n=2, axis=1)
    butterfly_violations = int((d2 < -1e-3).sum())
    total_points = int(iv_grid.size)
    return ArbitrageReport(
        calendar_violations=calendar_violations,
        butterfly_violations=butterfly_violations,
        total_points=total_points,
    )


def enforce_calendar_monotonicity(iv_grid: np.ndarray, expiry_axis: np.ndarray) -> np.ndarray:
    total_variance = (iv_grid**2) * expiry_axis[:, None]
    repaired_tv = np.zeros_like(total_variance)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for j in range(total_variance.shape[1]):
        repaired_tv[:, j] = iso.fit_transform(expiry_axis, total_variance[:, j])
    repaired_iv = np.sqrt(np.maximum(repaired_tv, 1e-10) / expiry_axis[:, None])
    return repaired_iv
