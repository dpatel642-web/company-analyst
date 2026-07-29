"""Phase B gate: the options math has to earn trust before anything is built on it.

Four independent kinds of evidence:
  1. published textbook values          (catches transcription errors)
  2. put-call parity                    (catches sign and discounting errors)
  3. agreement with a binomial lattice  (catches formula errors, shares no arithmetic)
  4. delta vs finite difference         (catches a wrong greek)
"""

from __future__ import annotations

import math

import pytest

from canalyst.options.bs import bs_delta, bs_price, intrinsic, strike_from_delta
from canalyst.options.crosscheck import crr_price

# ---------------------------------------------------------------- published values

# Textbook at-the-money case: S=K=100, T=1, r=5%, sigma=20%.
# Call 10.4506, Put 5.5735.
# Hull, Options Futures and Other Derivatives, example: S=42, K=40, r=10%,
# sigma=20%, T=0.5. Call 4.76, Put 0.81.
PUBLISHED = [
    # S,   K,    T,   r,    sigma, kind,   expected, tol
    (100.0, 100.0, 1.0, 0.05, 0.20, "call", 10.4506, 1e-4),
    (100.0, 100.0, 1.0, 0.05, 0.20, "put", 5.5735, 1e-4),
    (42.0, 40.0, 0.5, 0.10, 0.20, "call", 4.7594, 1e-3),
    (42.0, 40.0, 0.5, 0.10, 0.20, "put", 0.8086, 1e-3),
]


@pytest.mark.parametrize("S,K,T,r,sigma,kind,expected,tol", PUBLISHED)
def test_matches_published_values(S, K, T, r, sigma, kind, expected, tol):
    assert bs_price(S, K, T, r, sigma, kind) == pytest.approx(expected, abs=tol)


def test_atm_call_delta_is_n_d1():
    # d1 = (0 + (0.05 + 0.02)) / 0.20 = 0.35, N(0.35) = 0.636831
    assert bs_delta(100.0, 100.0, 1.0, 0.05, 0.20, "call") == pytest.approx(
        0.636831, abs=1e-5
    )


# ------------------------------------------------------------------ put-call parity

PARITY_CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
    (42.0, 40.0, 0.5, 0.10, 0.20, 0.0),
    (250.0, 300.0, 0.25, 0.04, 0.65, 0.0),  # deep OTM, TSLA-like vol
    (250.0, 180.0, 2.0, 0.045, 0.55, 0.0),  # deep ITM, long dated
    (100.0, 105.0, 1.0, 0.05, 0.30, 0.03),  # with a dividend yield
]


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARITY_CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    """C - P = S e^{-qT} - K e^{-rT}, exactly, for European options."""
    call = bs_price(S, K, T, r, sigma, "call", q=q)
    put = bs_price(S, K, T, r, sigma, "put", q=q)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARITY_CASES)
def test_delta_parity(S, K, T, r, sigma, q):
    """Call delta - put delta = e^{-qT}."""
    dc = bs_delta(S, K, T, r, sigma, "call", q=q)
    dp = bs_delta(S, K, T, r, sigma, "put", q=q)
    assert dc - dp == pytest.approx(math.exp(-q * T), abs=1e-10)


# -------------------------------------------------------- independent engine agreement


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARITY_CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_closed_form_agrees_with_binomial(S, K, T, r, sigma, q, kind):
    """Closed form vs a 2000-step CRR lattice. No shared arithmetic between them."""
    analytic = bs_price(S, K, T, r, sigma, kind, q=q)
    lattice = crr_price(S, K, T, r, sigma, kind, q=q, steps=2000)
    assert lattice == pytest.approx(analytic, abs=1e-2, rel=1e-3)


def test_binomial_converges_toward_closed_form():
    """More steps must mean less error, or the lattice is not actually converging."""
    args = (100.0, 100.0, 1.0, 0.05, 0.20, "call")
    analytic = bs_price(*args)
    errors = [
        abs(crr_price(*args, steps=n) - analytic) for n in (10, 50, 250, 1250)
    ]
    assert errors == sorted(errors, reverse=True), errors


def test_american_call_no_dividend_equals_european():
    """Without dividends, early exercise of a call is never optimal."""
    args = (100.0, 95.0, 1.0, 0.05, 0.30, "call")
    european = crr_price(*args, q=0.0, steps=1000, american=False)
    american = crr_price(*args, q=0.0, steps=1000, american=True)
    assert american == pytest.approx(european, abs=1e-6)


def test_american_put_is_worth_more_than_european():
    """Early exercise of a put has real value, so the American price must exceed it."""
    args = (100.0, 120.0, 1.0, 0.06, 0.25, "put")
    european = crr_price(*args, steps=1000, american=False)
    american = crr_price(*args, steps=1000, american=True)
    assert american > european + 1e-3


# ------------------------------------------------------------------- delta vs numeric


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARITY_CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_finite_difference(S, K, T, r, sigma, q, kind):
    h = S * 1e-5
    up = bs_price(S + h, K, T, r, sigma, kind, q=q)
    down = bs_price(S - h, K, T, r, sigma, kind, q=q)
    numeric = (up - down) / (2 * h)
    assert bs_delta(S, K, T, r, sigma, kind, q=q) == pytest.approx(numeric, abs=1e-6)


# ------------------------------------------------------------------ delta inversion


@pytest.mark.parametrize("target", [0.05, 0.15, 0.25, 0.40, 0.50, 0.75, 0.95])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_strike_from_delta_round_trips(target, kind):
    """The whole point of the inversion: feed the strike back and recover the delta."""
    S, T, r, sigma, q = 250.0, 21 / 252, 0.045, 0.60, 0.0
    K = strike_from_delta(S, T, r, sigma, target, kind, q=q)
    recovered = abs(bs_delta(S, K, T, r, sigma, kind, q=q))
    assert recovered == pytest.approx(target, abs=1e-9)


def test_lower_call_delta_means_higher_strike():
    """A 0.10-delta call must be further out of the money than a 0.40-delta call."""
    S, T, r, sigma = 250.0, 21 / 252, 0.045, 0.60
    far = strike_from_delta(S, T, r, sigma, 0.10, "call")
    near = strike_from_delta(S, T, r, sigma, 0.40, "call")
    assert far > near > 0


def test_strike_from_delta_round_trips_with_dividend():
    S, T, r, sigma, q = 100.0, 0.5, 0.04, 0.25, 0.035
    K = strike_from_delta(S, T, r, sigma, 0.25, "call", q=q)
    assert abs(bs_delta(S, K, T, r, sigma, "call", q=q)) == pytest.approx(
        0.25, abs=1e-9
    )


# ------------------------------------------------------------------------ edge cases


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K", [(110.0, 100.0), (90.0, 100.0), (100.0, 100.0)])
def test_expiry_collapses_to_intrinsic(S, K, kind):
    assert bs_price(S, K, 0.0, 0.05, 0.30, kind) == intrinsic(S, K, kind)
    assert bs_price(S, K, -1.0, 0.05, 0.30, kind) == intrinsic(S, K, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_vol_collapses_to_intrinsic(kind):
    """A zero-vol option is worth its intrinsic value. The engine relies on this to
    make a zero-premium covered call collapse exactly onto buy-and-hold."""
    assert bs_price(110.0, 100.0, 1.0, 0.05, 0.0, kind) == intrinsic(110.0, 100.0, kind)


def test_price_is_monotone_in_vol():
    prices = [bs_price(100.0, 100.0, 1.0, 0.05, s, "call") for s in (0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


def test_price_respects_no_arbitrage_bounds():
    """Call price sits between intrinsic-discounted and spot."""
    S, K, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 0.25
    call = bs_price(S, K, T, r, sigma, "call")
    lower = max(S - K * math.exp(-r * T), 0.0)
    assert lower <= call <= S


def test_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        bs_price(-1.0, 100.0, 1.0, 0.05, 0.2, "call")
    with pytest.raises(ValueError):
        bs_price(100.0, 0.0, 1.0, 0.05, 0.2, "call")
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        strike_from_delta(100.0, 1.0, 0.05, 0.2, 1.5, "call")
    with pytest.raises(ValueError):
        strike_from_delta(100.0, 1.0, 0.05, 0.2, 0.0, "call")
