from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class OptionsDataProvider(ABC):
    @abstractmethod
    def fetch_chain(
        self,
        ticker: str,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch options chain snapshot for a ticker."""

    @abstractmethod
    def fetch_spot(self, ticker: str) -> float:
        """Fetch spot price."""
