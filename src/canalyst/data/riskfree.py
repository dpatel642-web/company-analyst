"""Daily risk-free rate from the 13-week Treasury bill (^IRX).

The handout hardcodes 2% across a window in which the 3-month bill went from about
0.05% to about 5.4% and back toward 4%. That single constant is wrong in both
directions, and Sharpe is a direct function of it, so it is not a harmless
simplification. It also silently discards a real return: premium cash collected by a
call writer earns the prevailing short rate, and over 2022-2026 that is not noise.

^IRX is quoted in percent on a bank-discount basis. Converting exactly would require
each bill's days to maturity; instead the percent is read as an annualised simple rate
and converted to continuous compounding via ln(1+r). At 4% that approximation is worth
roughly 8 basis points, against the several-hundred-basis-point error it replaces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

IRX_TICKER = "^IRX"
#: Used only to fill leading gaps before the first observation.
FALLBACK_RATE = 0.02


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
) -> pd.Series:
    """Continuously-compounded daily risk-free rate aligned to `index`.

    Forward-fills across days the bill did not print, which is the right behaviour for
    a rate: the last observed rate is still the prevailing rate. Raises when `strict`
    and the fetch fails, because silently falling back to a constant is precisely the
    error this module exists to remove.
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
        return pd.Series(
            np.log1p(FALLBACK_RATE), index=index, name="rf_continuous"
        )

    aligned = simple.reindex(simple.index.union(index)).ffill().reindex(index)
    aligned = aligned.bfill()  # leading gap before the first print
    if aligned.isna().any():
        aligned = aligned.fillna(FALLBACK_RATE)
    out = continuous_from_simple(aligned)
    out.name = "rf_continuous"
    return out


def annualised_mean(rf_continuous: pd.Series) -> float:
    """Average continuously-compounded rate over the window, for reporting."""
    return float(rf_continuous.mean())
