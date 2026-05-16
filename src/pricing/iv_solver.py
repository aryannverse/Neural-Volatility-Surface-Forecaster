from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq

from src.pricing.black_scholes import OptionType, bs_price, greeks


@dataclass(frozen=True)
class IVSolverConfig:
    tol: float = 1e-7
    max_iter: int = 100
    sigma_low: float = 1e-6
    sigma_high: float = 5.0
    fallback_to_brent: bool = True
    n_jobs: int = -1


def implied_volatility(
    market_price: float,
    spot_price: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType | str,
    config: IVSolverConfig | None = None,
) -> float:
    """
    Implied volatility inversion:
      1) Newton-Raphson on f(sigma)=BS(sigma)-market
      2) Brent fallback for guaranteed bracketed root search
    """
    cfg = config or IVSolverConfig()
    otype = OptionType(option_type)
    target = max(market_price, 1e-10)
    sigma = 0.2

    for _ in range(cfg.max_iter):
        price = bs_price(spot_price, strike, time_to_expiry, risk_free_rate, sigma, otype)
        diff = price - target
        if abs(diff) < cfg.tol:
            return float(max(sigma, cfg.sigma_low))

        vega = greeks(
            spot=spot_price,
            strike=strike,
            maturity=time_to_expiry,
            rate=risk_free_rate,
            vol=sigma,
            option_type=otype,
        ).vega
        if abs(vega) < 1e-10:
            break
        sigma -= diff / vega
        sigma = float(np.clip(sigma, cfg.sigma_low, cfg.sigma_high))

    if not cfg.fallback_to_brent:
        return float("nan")

    def objective(sig: float) -> float:
        return (
            bs_price(
                spot=spot_price,
                strike=strike,
                maturity=time_to_expiry,
                rate=risk_free_rate,
                vol=sig,
                option_type=otype,
            )
            - target
        )

    try:
        root = brentq(objective, cfg.sigma_low, cfg.sigma_high, xtol=cfg.tol)
        return float(root)
    except ValueError:
        return float("nan")


def implied_volatility_vectorized(
    market_price: Iterable[float],
    spot_price: Iterable[float],
    strike: Iterable[float],
    time_to_expiry: Iterable[float],
    risk_free_rate: Iterable[float] | float,
    option_type: Iterable[str],
    config: IVSolverConfig | None = None,
) -> np.ndarray:
    cfg = config or IVSolverConfig()
    mp = np.asarray(list(market_price), dtype=float)
    sp = np.asarray(list(spot_price), dtype=float)
    kk = np.asarray(list(strike), dtype=float)
    tt = np.asarray(list(time_to_expiry), dtype=float)
    rr = (
        np.full_like(mp, fill_value=float(risk_free_rate), dtype=float)
        if np.isscalar(risk_free_rate)
        else np.asarray(list(risk_free_rate), dtype=float)
    )
    oo = np.asarray(list(option_type), dtype=str)

    outputs = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(implied_volatility)(
            market_price=float(mp[i]),
            spot_price=float(sp[i]),
            strike=float(kk[i]),
            time_to_expiry=float(tt[i]),
            risk_free_rate=float(rr[i]),
            option_type=oo[i],
            config=cfg,
        )
        for i in range(len(mp))
    )
    return np.asarray(outputs, dtype=float)
