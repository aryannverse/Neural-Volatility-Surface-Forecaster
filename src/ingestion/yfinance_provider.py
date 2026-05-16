from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.ingestion.base import OptionsDataProvider
from src.pricing import implied_volatility_vectorized
from src.pricing.black_scholes import greeks
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class YFinanceProviderConfig:
    cache_dir: Path
    risk_free_rate: float = 0.03


class YFinanceOptionsProvider(OptionsDataProvider):
    """
    Free market-data backend. This prioritizes consistency and resilience:
    - Pull all available expiries from Yahoo
    - Preserve raw chain fields
    - Compute mid-price, maturity, moneyness
    - Backfill IV and Greeks when absent
    """

    def __init__(self, config: YFinanceProviderConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_spot(self, ticker: str) -> float:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        last = info.get("lastPrice") or info.get("regularMarketPrice")
        if last is None:
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                raise ValueError(f"Unable to fetch spot price for {ticker}")
            last = float(hist["Close"].iloc[-1])
        return float(last)

    def fetch_chain(
        self,
        ticker: str,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        as_of_ts = as_of or datetime.now(timezone.utc)
        cache_path = self.config.cache_dir / f"{ticker}_{as_of_ts:%Y%m%d}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        tk = yf.Ticker(ticker)
        spot = self.fetch_spot(ticker)
        rows: list[pd.DataFrame] = []
        expiries = list(tk.options)
        if not expiries:
            raise ValueError(f"No listed option expiries found for {ticker}")

        for expiry_str in expiries:
            chain = tk.option_chain(expiry_str)
            expiry = pd.Timestamp(expiry_str).tz_localize("UTC")
            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                local = df.copy()
                local["option_type"] = opt_type
                local["expiry"] = expiry
                local["ticker"] = ticker.upper()
                local["spot"] = spot
                local["timestamp"] = pd.Timestamp(as_of_ts)
                rows.append(local)

        chain_df = pd.concat(rows, ignore_index=True)
        chain_df.rename(
            columns={
                "openInterest": "open_interest",
                "impliedVolatility": "iv_yahoo",
                "lastTradeDate": "last_trade_date",
                "lastPrice": "last_price",
            },
            inplace=True,
        )
        if "iv_yahoo" not in chain_df.columns:
            chain_df["iv_yahoo"] = np.nan
        chain_df["bid"] = pd.to_numeric(chain_df["bid"], errors="coerce")
        chain_df["ask"] = pd.to_numeric(chain_df["ask"], errors="coerce")
        chain_df["strike"] = pd.to_numeric(chain_df["strike"], errors="coerce")
        chain_df["mid_price"] = (chain_df["bid"] + chain_df["ask"]) / 2.0
        chain_df["mid_price"] = chain_df["mid_price"].where(chain_df["mid_price"] > 0.0, chain_df["last_price"])
        chain_df["ttm"] = (
            (chain_df["expiry"] - chain_df["timestamp"]).dt.total_seconds()
            / (365.25 * 24 * 3600)
        )
        chain_df["ttm"] = chain_df["ttm"].clip(lower=1.0 / 365.25)
        chain_df["moneyness"] = chain_df["strike"] / chain_df["spot"]

        valid_iv = chain_df["iv_yahoo"].astype(float).where(chain_df["iv_yahoo"].astype(float) > 1e-5)
        needs_compute = valid_iv.isna() | ~np.isfinite(valid_iv)
        if needs_compute.any():
            iv_calc = implied_volatility_vectorized(
                market_price=chain_df.loc[needs_compute, "mid_price"].fillna(0.0),
                spot_price=chain_df.loc[needs_compute, "spot"],
                strike=chain_df.loc[needs_compute, "strike"],
                time_to_expiry=chain_df.loc[needs_compute, "ttm"],
                risk_free_rate=self.config.risk_free_rate,
                option_type=chain_df.loc[needs_compute, "option_type"],
            )
            chain_df.loc[needs_compute, "iv"] = iv_calc

        if "iv" not in chain_df.columns:
            chain_df["iv"] = np.nan
        chain_df["iv"] = chain_df["iv"].fillna(valid_iv).astype(float)

        for greek in ["delta", "gamma", "theta", "vega", "rho"]:
            chain_df[greek] = np.nan

        greeks_needed = chain_df["iv"].notna() & np.isfinite(chain_df["iv"])
        g_idx = chain_df.index[greeks_needed]
        for idx in g_idx:
            row = chain_df.loc[idx]
            try:
                g = greeks(
                    spot=float(row["spot"]),
                    strike=float(row["strike"]),
                    maturity=float(row["ttm"]),
                    rate=self.config.risk_free_rate,
                    vol=float(row["iv"]),
                    option_type=str(row["option_type"]),
                )
            except Exception:
                continue
            chain_df.at[idx, "delta"] = g.delta
            chain_df.at[idx, "gamma"] = g.gamma
            chain_df.at[idx, "theta"] = g.theta
            chain_df.at[idx, "vega"] = g.vega
            chain_df.at[idx, "rho"] = g.rho

        chain_df.to_parquet(cache_path, index=False)
        LOGGER.info("Saved options chain cache %s rows=%d", cache_path.name, len(chain_df))
        return chain_df
