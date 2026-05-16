from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np
from scipy.stats import norm

try:
    from py_vollib.black_scholes_merton import black_scholes_merton as vollib_bsm
except Exception:  # pragma: no cover
    vollib_bsm = None


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Greeks(NamedTuple):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _safe_inputs(
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
) -> tuple[float, float, float, float]:
    return max(spot, 1e-12), max(strike, 1e-12), max(maturity, 1e-10), max(vol, 1e-8)


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
) -> tuple[float, float]:
    s, k, t, sigma = _safe_inputs(spot, strike, maturity, vol)
    d1 = (np.log(s / k) + (rate + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(d1), float(d2)


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: OptionType | str,
) -> float:
    """
    Black-Scholes-Merton price under risk-neutral dynamics.

    PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0
    """
    otype = OptionType(option_type)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol)
    discounted_k = strike * np.exp(-rate * maturity)
    if otype == OptionType.CALL:
        return float(spot * norm.cdf(d1) - discounted_k * norm.cdf(d2))
    return float(discounted_k * norm.cdf(-d2) - spot * norm.cdf(-d1))


def bs_price_vollib(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: OptionType | str,
    dividend_yield: float = 0.0,
) -> float:
    if vollib_bsm is None:
        return bs_price(spot, strike, maturity, rate, vol, option_type)
    flag = "c" if OptionType(option_type) == OptionType.CALL else "p"
    return float(vollib_bsm(flag, spot, strike, maturity, rate, vol, dividend_yield))


def bs_price_vectorized(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray | float,
    vol: np.ndarray,
    option_type: np.ndarray,
) -> np.ndarray:
    vec_price = np.vectorize(bs_price)
    return vec_price(spot, strike, maturity, rate, vol, option_type)


def greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: OptionType | str,
) -> Greeks:
    otype = OptionType(option_type)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol)
    s, _, t, sigma = _safe_inputs(spot, strike, maturity, vol)
    sqrt_t = np.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-rate * t)

    delta = norm.cdf(d1) if otype == OptionType.CALL else norm.cdf(d1) - 1.0
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    vega = s * pdf_d1 * sqrt_t

    if otype == OptionType.CALL:
        theta = (
            -(s * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - rate * strike * discount * norm.cdf(d2)
        )
        rho = strike * t * discount * norm.cdf(d2)
    else:
        theta = (
            -(s * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + rate * strike * discount * norm.cdf(-d2)
        )
        rho = -strike * t * discount * norm.cdf(-d2)

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho),
    )
