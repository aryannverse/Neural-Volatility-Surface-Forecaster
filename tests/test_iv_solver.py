from __future__ import annotations

import numpy as np

from src.pricing.black_scholes import bs_price
from src.pricing.iv_solver import implied_volatility, implied_volatility_vectorized


def test_implied_volatility_recovers_true_sigma() -> None:
    sigma_true = 0.25
    price = bs_price(spot=100.0, strike=105.0, maturity=0.5, rate=0.03, vol=sigma_true, option_type="call")
    sigma_hat = implied_volatility(
        market_price=price,
        spot_price=100.0,
        strike=105.0,
        time_to_expiry=0.5,
        risk_free_rate=0.03,
        option_type="call",
    )
    assert abs(sigma_hat - sigma_true) < 1e-3


def test_vectorized_iv_solver() -> None:
    sigmas = np.array([0.15, 0.2, 0.3])
    prices = np.array(
        [
            bs_price(100.0, 90.0, 0.25, 0.02, sigmas[0], "call"),
            bs_price(100.0, 100.0, 0.5, 0.02, sigmas[1], "call"),
            bs_price(100.0, 110.0, 1.0, 0.02, sigmas[2], "put"),
        ]
    )
    out = implied_volatility_vectorized(
        market_price=prices,
        spot_price=np.array([100.0, 100.0, 100.0]),
        strike=np.array([90.0, 100.0, 110.0]),
        time_to_expiry=np.array([0.25, 0.5, 1.0]),
        risk_free_rate=0.02,
        option_type=np.array(["call", "call", "put"]),
    )
    assert np.allclose(out, sigmas, atol=1e-3)
