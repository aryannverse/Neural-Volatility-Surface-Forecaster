from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SurfaceKey:
    ticker: str
    timestamp: datetime
    grid_kind: str = "moneyness-expiry"
