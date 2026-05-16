from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from src.ingestion.base import OptionsDataProvider


@dataclass(frozen=True)
class PolygonProviderConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"


class PolygonOptionsProvider(OptionsDataProvider):
    """
    Optional provider for users with Polygon access.
    This module is not required for free-mode operation.
    """

    def __init__(self, config: PolygonProviderConfig):
        self.config = config

    def fetch_spot(self, ticker: str) -> float:
        url = f"{self.config.base_url}/v2/aggs/ticker/{ticker.upper()}/prev"
        r = requests.get(url, params={"adjusted": "true", "apiKey": self.config.api_key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            raise ValueError(f"No spot result for {ticker}")
        return float(data["results"][0]["c"])

    def fetch_chain(self, ticker: str, as_of: datetime | None = None) -> pd.DataFrame:
        date = (as_of or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        url = f"{self.config.base_url}/v3/snapshot/options/{ticker.upper()}"
        params = {"apiKey": self.config.api_key, "limit": 1000}
        frames: list[dict] = []
        while True:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            for row in payload.get("results", []):
                d = row.get("details", {})
                q = row.get("last_quote", {})
                g = row.get("greeks", {})
                iv = row.get("implied_volatility")
                frames.append(
                    {
                        "ticker": ticker.upper(),
                        "timestamp": date,
                        "expiry": d.get("expiration_date"),
                        "option_type": d.get("contract_type"),
                        "strike": d.get("strike_price"),
                        "bid": q.get("bid"),
                        "ask": q.get("ask"),
                        "mid_price": None if q.get("bid") is None or q.get("ask") is None else (q["bid"] + q["ask"]) / 2.0,
                        "open_interest": row.get("open_interest"),
                        "volume": row.get("day", {}).get("volume"),
                        "iv": iv,
                        "delta": g.get("delta"),
                        "gamma": g.get("gamma"),
                        "theta": g.get("theta"),
                        "vega": g.get("vega"),
                    }
                )
            next_url = payload.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {"apiKey": self.config.api_key}
        if not frames:
            raise ValueError(f"No options chain rows returned for {ticker}")
        return pd.DataFrame(frames)
