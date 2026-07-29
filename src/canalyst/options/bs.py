"""Black-Scholes-Merton pricing for European options, with continuous dividend yield.

Conventions used throughout the project:
  S     spot, in the same (unadjusted) price units the strike is struck on
  K     strike
  T     time to expiry in years, measured in trading days / 252
  r     continuously-compounded risk-free rate, annualised
  sigma annualised volatility
  q     continuous dividend yield, annualised

Every function degrades to intrinsic value when T <= 0 or sigma <= 0 rather than
returning NaN, so an expiring or zero-vol position still marks correctly.
"""

from __future__ import annotations

import math
from typing import Literal

from ..normal import cdf, inv_cdf

Kind = Literal["call", "put"]

_MIN_SIGMA = 1e-12
_MIN_T = 1e-12
#: Minimum time value, as a fraction of spot, for volatility to be identifiable from a
#: price. Below this, vega has collapsed and the inversion is ill-conditioned.
MIN_TIME_VALUE_FRACTION = 1e-4


def _validate(S: float, K: float, kind: str) -> None:
    if S <= 0:
        raise ValueError(f"spot must be positive, got {S!r}")
    if K <= 0:
        raise ValueError(f"strike must be positive, got {K!r}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def intrinsic(S: float, K: float, kind: Kind) -> float:
    """Payoff if exercised right now."""
    _validate(S, K, kind)
    return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: Kind,
    q: float = 0.0,
) -> float:
    """European option price. Falls back to intrinsic at T <= 0 or sigma <= 0."""
    _validate(S, K, kind)
    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        return intrinsic(S, K, kind)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    spot_pv = S * math.exp(-q * T)
    strike_pv = K * math.exp(-r * T)

    if kind == "call":
        return spot_pv * cdf(d1) - strike_pv * cdf(d2)
    return strike_pv * cdf(-d2) - spot_pv * cdf(-d1)


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: Kind,
    q: float = 0.0,
) -> float:
    """dPrice/dSpot. Call delta is positive, put delta negative.

    At expiry this is the step function: 1/0 for a call, -1/0 for a put. Exactly
    at the money it returns 0.5 (or -0.5), an arbitrary but harmless tie-break.
    """
    _validate(S, K, kind)
    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        if math.isclose(S, K):
            return 0.5 if kind == "call" else -0.5
        if kind == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    discount = math.exp(-q * T)
    if kind == "call":
        return discount * cdf(d1)
    return -discount * cdf(-d1)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: Kind,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float | None:
    """Invert Black-Scholes for volatility. Returns None when no solution exists.

    This is what turns a real traded option price into a measurable implied volatility,
    and therefore what lets the variance risk premium be measured instead of assumed.

    Bisection rather than Newton. Newton is faster but vega collapses toward zero for
    deep out-of-the-money and near-expiry contracts, which is exactly the population being
    inverted here, and a vanishing derivative makes Newton diverge silently. Bisection
    cannot diverge; it either brackets a root or reports that none exists.

    Returns None rather than raising, and rather than clamping to a bound, for prices that
    violate no-arbitrage (below intrinsic, or above the maximum an option can be worth).
    Those occur in real data from stale quotes and wide spreads, and a clamped 0.001 or
    5.0 would enter a mean as if it were a measurement.
    """
    _validate(S, K, kind)
    if T <= _MIN_T or price is None or price <= 0 or not math.isfinite(price):
        return None

    lower_bound = intrinsic(S * math.exp(-q * T), K * math.exp(-r * T), kind)
    upper_bound = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    if price < lower_bound - tol or price > upper_bound + tol:
        return None  # outside no-arbitrage bounds

    # Refuse when there is no volatility information in the price. A deep in-the-money
    # contract is worth its intrinsic value almost exactly, vega collapses, and the
    # inversion becomes ill-conditioned: a one-cent price difference moves the answer by
    # tens of vol points. Returning a number there implies precision that does not exist,
    # and since these prices are being averaged into a measured variance risk premium, one
    # garbage inversion contaminates the estimate. Absence of a measurement is the truth.
    time_value = price - lower_bound
    if time_value < MIN_TIME_VALUE_FRACTION * S:
        return None

    lo, hi = 1e-6, 5.0
    if bs_price(S, K, T, r, hi, kind, q=q) < price:
        return None  # not attainable even at 500% vol

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, kind, q=q) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def strike_from_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    kind: Kind,
    q: float = 0.0,
) -> float:
    """Invert delta to a strike. `target_delta` is a magnitude in (0, 1).

    Call delta = e^{-qT} N(d1), so N(d1) = target * e^{qT} and d1 = inv_cdf(that).
    Put delta magnitude = e^{-qT} N(-d1), so d1 = -inv_cdf(target * e^{qT}).
    Then from d1's definition:
        ln(S/K) = d1 * sigma * sqrt(T) - (r - q + sigma^2/2) T
        K       = S * exp(-(d1 * sigma * sqrt(T)) + (r - q + sigma^2/2) T)

    A deep-dated, high-yield combination can push target * e^{qT} to >= 1, which has
    no solution; inv_cdf raises rather than clamping.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta!r}")
    if S <= 0:
        raise ValueError(f"spot must be positive, got {S!r}")
    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        raise ValueError("strike_from_delta needs positive T and sigma")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    p = target_delta * math.exp(q * T)
    d1 = inv_cdf(p) if kind == "call" else -inv_cdf(p)
    drift = (r - q + 0.5 * sigma * sigma) * T
    return S * math.exp(-(d1 * sigma * math.sqrt(T)) + drift)
