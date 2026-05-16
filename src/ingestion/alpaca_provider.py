from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests

from src.ingestion.base import OptionsDataProvider


@dataclass(frozen=True)
class AlpacaProviderConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://data.alpaca.markets"


class AlpacaOptionsProvider(OptionsDataProvider):
    """
    Optional account-based provider. Free mode does not require Alpaca.
    """

    def __init__(self, config: AlpacaProviderConfig):
        self.config = config

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.api_secret,
        }

    def fetch_spot(self, ticker: str) -> float:
        url = f"{self.config.base_url}/v2/stocks/{ticker.upper()}/quotes/latest"
        r = requests.get(url, headers=self._headers, timeout=20)
        r.raise_for_status()
        quote = r.json().get("quote", {})
        bid = quote.get("bp")
        ask = quote.get("ap")
        if bid is None or ask is None:
            raise ValueError(f"No valid latest quote returned for {ticker}")
        return float((bid + ask) / 2.0)

    def fetch_chain(self, ticker: str, as_of: datetime | None = None) -> pd.DataFrame:
        date = (as_of or datetime.utcnow()).strftime("%Y-%m-%d")
        url = f"{self.config.base_url}/v1beta1/options/snapshots/{ticker.upper()}"
        r = requests.get(url, headers=self._headers, params={"feed": "indicative"}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows: list[dict] = []
        for sym, row in payload.get("snapshots", {}).items():
            latest = row.get("latestQuote", {})
            greeks = row.get("greeks", {})
            details = row.get("details", {})
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "contract_symbol": sym,
                    "timestamp": date,
                    "expiry": details.get("expiration_date"),
                    "option_type": details.get("type"),
                    "strike": details.get("strike_price"),
                    "bid": latest.get("bp"),
                    "ask": latest.get("ap"),
                    "mid_price": None
                    if latest.get("bp") is None or latest.get("ap") is None
                    else (latest["bp"] + latest["ap"]) / 2.0,
                    "iv": row.get("impliedVolatility"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "open_interest": row.get("open_interest"),
                    "volume": row.get("volume"),
                }
            )
        if not rows:
            raise ValueError(f"No Alpaca options rows for {ticker}")
        return pd.DataFrame(rows)
