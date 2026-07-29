"""Realised volatility estimators, and the knob that admits they are not implied vol.

No free source carries a five-year history of option chains, so the backtest prices
options off realised volatility. That is not a neutral choice. Implied vol trades
above realised vol on average (the variance risk premium is why writing options pays
at all), so realised-vol pricing systematically *understates* the premium a call
writer would really have collected. The base case is therefore conservative, and
`iv_markup` exists to measure how much that conservatism costs.

Never report a marked-up number as the headline. It is a sensitivity, not a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def close_to_close(
    close: pd.Series, window: int = 60, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Annualised rolling stdev of log returns.

    Log returns, not simple returns: the estimator assumes the quantity being
    averaged is additive over time, which only log returns are. The handout uses
    `pct_change`, which is close enough at daily frequency to be invisible for most
    tickers and not invisible for a 40%-vol one.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(trading_days)


def yang_zhang(
    ohlc: pd.DataFrame, window: int = 60, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Yang-Zhang estimator: drift-independent and handles overnight gaps.

    Close-to-close throws away the intraday range and charges the full overnight gap
    to volatility. Yang-Zhang splits variance into overnight, open-to-close, and a
    Rogers-Satchell term, which matters for a ticker that habitually gaps.

    Requires Open/High/Low/Close columns.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    missing = {"Open", "High", "Low", "Close"} - set(ohlc.columns)
    if missing:
        raise ValueError(f"yang_zhang needs OHLC, missing {sorted(missing)}")

    o, h, l, c = ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = c.shift(1)

    overnight = np.log(o / prev_close)
    open_to_close = np.log(c / o)
    rogers_satchell = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    var_overnight = overnight.rolling(window).var()
    var_open_close = open_to_close.rolling(window).var()
    var_rs = rogers_satchell.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    total = var_overnight + k * var_open_close + (1.0 - k) * var_rs
    return np.sqrt(total.clip(lower=0.0)) * np.sqrt(trading_days)


def apply_markup(sigma: pd.Series | float, iv_markup: float = 1.0):
    """Scale realised vol toward implied. 1.0 is the headline; 1.1-1.2 is sensitivity."""
    if iv_markup <= 0:
        raise ValueError(f"iv_markup must be positive, got {iv_markup!r}")
    return sigma * iv_markup
