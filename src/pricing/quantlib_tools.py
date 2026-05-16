from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    import QuantLib as ql
except Exception:  # pragma: no cover
    ql = None


@dataclass(frozen=True)
class CurvePoint:
    tenor_days: int
    rate: float


def build_flat_curve(rate: float, valuation_date: datetime | None = None):
    if ql is None:
        return None
    dt = valuation_date or datetime.utcnow()
    ql_date = ql.Date(dt.day, dt.month, dt.year)
    ql.Settings.instance().evaluationDate = ql_date
    day_counter = ql.Actual365Fixed()
    return ql.YieldTermStructureHandle(ql.FlatForward(ql_date, rate, day_counter))


def quantlib_black_price(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    if ql is None:
        raise RuntimeError("QuantLib is not available in the environment.")

    payoff_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(payoff_type, strike)

    today = ql.Settings.instance().evaluationDate
    maturity = today + int(np.round(maturity_years * 365))
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_curve = build_flat_curve(rate)
    q_curve = build_flat_curve(0.0)
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    process = ql.BlackScholesMertonProcess(spot_handle, q_curve, r_curve, vol_ts)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return float(option.NPV())
