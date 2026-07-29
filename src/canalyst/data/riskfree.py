"""Daily risk-free rate from the 13-week Treasury bill (^IRX).

The handout hardcodes 2% across a window in which the 3-month bill went from about
0.05% to about 5.4% and back toward 4%. That single constant is wrong in both
directions, and Sharpe is a direct function of it, so it is not a harmless
simplification. It also silently discards a real return: premium cash collected by a
call writer earns the prevailing short rate, and over 2022-2026 that is not noise.

^IRX is quoted in percent on a bank-discount basis. Converting exactly would require each
bill's days to maturity; instead the percent is read as an annualised simple rate and
converted to continuous compounding via ln(1+r).

That approximation is worth **15.4bp at 4% and 25.3bp at the window's 5.4% peak**, not the
8bp an earlier version of this docstring claimed, and it is negative on every single day
because two approximations push the same way: skipping the discount-to-yield conversion,
and 360 against 365 day count. Over 2021-2026 the mean lands at 3.54% against a true
continuously-compounded bond-equivalent 3.69%. A one-signed rf error biases Sharpe upward
by roughly 0.015 at 10% volatility.

The choice is still defensible, since it replaces a several-hundred-basis-point error. The
understated magnitude was not, in the one file whose whole argument is that rf precision is
not a harmless simplification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

IRX_TICKER = "^IRX"
#: Used only to fill leading gaps before the first observation.
FALLBACK_RATE = 0.02
#: A 13-week bill outside this range is a scaling error, not a rate. Slightly negative is
#: allowed: it has happened in other currencies and briefly in USD bill quotes.
MIN_PLAUSIBLE_RATE = -0.01
MAX_PLAUSIBLE_RATE = 0.25
#: How long one print may speak for. Beyond this the value is carried, not observed, and
#: an unbounded carry is how a single observation became a five-year constant.
MAX_CARRY_DAYS = 7
#: Minimum fraction of the window that must sit within MAX_CARRY_DAYS of a real print.
MIN_COVERAGE = 0.80


def fetch_irx(start, end) -> pd.Series | None:
    """Raw ^IRX closes as a decimal simple rate, or None if unavailable."""
    end_exclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    try:
        raw = yf.download(
            IRX_TICKER,
            start=pd.Timestamp(start).normalize(),
            end=end_exclusive,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
    if "Close" not in raw.columns:
        return None

    idx = pd.DatetimeIndex(raw.index)
    raw.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    series = raw["Close"].astype(float).dropna() / 100.0
    series.name = "rf_simple"
    return series if len(series) else None


def continuous_from_simple(simple: pd.Series) -> pd.Series:
    """ln(1 + r). Black-Scholes wants a continuously-compounded rate."""
    return np.log1p(simple)


def risk_free_series(
    index: pd.DatetimeIndex,
    start=None,
    end=None,
    strict: bool = True,
    min_coverage: float = MIN_COVERAGE,
    max_carry_days: int = MAX_CARRY_DAYS,
    return_coverage: bool = False,
):
    """Continuously-compounded daily risk-free rate aligned to `index`.

    Forward-fills across days the bill did not print, which is the right behaviour for a
    rate: the last observed rate is still the prevailing rate. But the fill is now
    BOUNDED, and `strict` guards a short response as well as a missing one.

    The hole this closes: `strict=True` used to guard only a total fetch failure.
    `fetch_irx` accepts any series of length >= 1, and an unbounded ffill then smeared it
    across the whole index. A response covering only 2021 pinned the rate at 0.05% for
    five years against a true mean of 3.54%, a 3.5pp one-signed error that overstates
    Sharpe by roughly 0.35 at 10% volatility and on its own flips a `sharpe >= 1` verdict.
    One observation became a five-year constant with no warning, in the module whose whole
    argument is that a silently substituted constant is the error to remove.

    Also bounds-checks the level. A mis-scaled print, 540 read as 5.40, yields a 186%
    continuous rate, and nothing downstream would question it.

    With `return_coverage`, also returns a boolean Series marking which index dates had a
    real print within `max_carry_days`, so the data-quality report can show how much of
    the series is carried rather than observed.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        raise ValueError("cannot build a risk-free series for an empty index")

    simple = fetch_irx(start or index[0], end or index[-1])
    if simple is None:
        if strict:
            raise RuntimeError(
                "could not fetch ^IRX; refusing to substitute a constant rate. "
                "Pass strict=False to accept the fallback explicitly."
            )
        out = pd.Series(np.log1p(FALLBACK_RATE), index=index, name="rf_continuous")
        return (out, pd.Series(False, index=index)) if return_coverage else out

    out_of_bounds = ((simple < MIN_PLAUSIBLE_RATE) | (simple > MAX_PLAUSIBLE_RATE)).sum()
    if out_of_bounds:
        message = (
            f"^IRX returned {out_of_bounds} observation(s) outside "
            f"[{MIN_PLAUSIBLE_RATE:.1%}, {MAX_PLAUSIBLE_RATE:.1%}], which is the "
            "signature of a scaling error rather than a real rate"
        )
        if strict:
            raise RuntimeError(message)
        simple = simple[
            (simple >= MIN_PLAUSIBLE_RATE) & (simple <= MAX_PLAUSIBLE_RATE)
        ]
        if simple.empty:
            raise RuntimeError(message + "; nothing plausible remains")

    # Bounded fill: a print is only allowed to speak for the next `max_carry_days`.
    combined = simple.reindex(simple.index.union(index)).ffill()
    stamps = pd.Series(combined.index, index=combined.index).where(
        combined.index.isin(simple.index)
    ).ffill()
    age_days = (combined.index.to_series() - stamps).dt.days
    fresh = (age_days <= max_carry_days).reindex(index).fillna(False)

    aligned = combined.reindex(index)
    had_print = pd.Series(index.isin(simple.index), index=index)

    coverage = float(fresh.mean())
    if coverage < min_coverage:
        message = (
            f"^IRX covers only {coverage:.0%} of the requested window within "
            f"{max_carry_days} days of a real print (needed {min_coverage:.0%}). "
            f"Got {len(simple)} observation(s) spanning "
            f"{simple.index[0].date()}..{simple.index[-1].date()} for an index spanning "
            f"{index[0].date()}..{index[-1].date()}. Refusing to smear that across the "
            "whole window; a short response is as dangerous as a missing one."
        )
        if strict:
            raise RuntimeError(message)

    aligned = aligned.bfill()  # leading gap before the first print
    if aligned.isna().any():
        aligned = aligned.fillna(FALLBACK_RATE)
    out = continuous_from_simple(aligned)
    out.name = "rf_continuous"
    return (out, had_print) if return_coverage else out


def annualised_mean(rf_continuous: pd.Series) -> float:
    """Average continuously-compounded rate over the window, for reporting."""
    return float(rf_continuous.mean())
