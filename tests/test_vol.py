"""Volatility estimators. Synthetic data with a known answer, so the test can fail."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canalyst.options.vol import apply_markup, close_to_close, yang_zhang

TRUE_SIGMA = 0.40
N_DAYS = 2000


def _gbm_closes(sigma: float = TRUE_SIGMA, n: int = N_DAYS, seed: int = 7) -> pd.Series:
    """Geometric Brownian motion with a known annualised vol."""
    rng = np.random.default_rng(seed)
    daily = sigma / np.sqrt(252)
    log_ret = rng.normal(0.0, daily, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="Close")


def _ohlc_from_closes(close: pd.Series, seed: int = 11) -> pd.DataFrame:
    """Wrap a close series in plausible OHLC that always brackets open and close."""
    rng = np.random.default_rng(seed)
    n = len(close)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(
        {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=close.index
    )


def test_close_to_close_recovers_known_vol():
    """Estimated vol must land near the vol actually used to generate the series.

    Sampling error on a 252-day window is roughly sigma/sqrt(2*252) = 0.018, so a
    0.06 tolerance is about three standard errors.
    """
    est = close_to_close(_gbm_closes(), window=252).dropna()
    assert est.mean() == pytest.approx(TRUE_SIGMA, abs=0.06)


def test_close_to_close_scales_with_true_vol():
    low = close_to_close(_gbm_closes(sigma=0.20), window=252).dropna().mean()
    high = close_to_close(_gbm_closes(sigma=0.80), window=252).dropna().mean()
    assert high > low * 2.5


def test_close_to_close_warmup_is_nan_not_zero():
    """A vol that could not be computed must be absent, never silently zero."""
    est = close_to_close(_gbm_closes(n=100), window=60)
    assert est.iloc[:60].isna().all()
    assert est.iloc[60:].notna().all()


def test_yang_zhang_is_in_a_sane_range():
    ohlc = _ohlc_from_closes(_gbm_closes())
    est = yang_zhang(ohlc, window=252).dropna()
    assert (est > 0).all()
    # Intraday range adds variance the close-to-close estimator cannot see, so YZ
    # should be at least as large, but still the same order of magnitude.
    assert TRUE_SIGMA * 0.7 < est.mean() < TRUE_SIGMA * 2.0


def test_yang_zhang_requires_ohlc():
    close_only = _gbm_closes().to_frame()
    with pytest.raises(ValueError, match="missing"):
        yang_zhang(close_only)


@pytest.mark.parametrize("bad_window", [0, 1, -5])
def test_estimators_reject_degenerate_windows(bad_window):
    with pytest.raises(ValueError):
        close_to_close(_gbm_closes(n=50), window=bad_window)
    with pytest.raises(ValueError):
        yang_zhang(_ohlc_from_closes(_gbm_closes(n=50)), window=bad_window)


def test_apply_markup_scales_and_defaults_to_identity():
    sigma = pd.Series([0.4, 0.5, 0.6])
    pd.testing.assert_series_equal(apply_markup(sigma, 1.0), sigma)
    assert apply_markup(0.40, 1.20) == pytest.approx(0.48)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_apply_markup_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        apply_markup(0.4, bad)
