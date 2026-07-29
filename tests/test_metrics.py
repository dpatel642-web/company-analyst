"""Performance statistics, checked against closed-form answers where they exist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canalyst.metrics import TRADING_DAYS, comparison_table, daily_simple_rf, summarise


def _curve(daily_return: float, n: int = TRADING_DAYS * 2, start: float = 100.0):
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(start * (1.0 + daily_return) ** np.arange(n), index=idx)


def test_constant_growth_has_exact_known_answers():
    """A deterministic curve pins CAGR and vol without any statistics."""
    daily = 0.0004
    value = _curve(daily, n=TRADING_DAYS * 3)
    perf = summarise(value, label="det")

    perf.verify()
    assert perf.annual_vol == pytest.approx(0.0, abs=1e-12)
    assert perf.cagr == pytest.approx((1 + daily) ** TRADING_DAYS - 1, rel=1e-6)
    assert perf.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_cumulative_return_agrees_between_methods():
    """Terminal-over-initial vs compounded daily log returns. Different arithmetic."""
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2021-07-29", periods=900)
    value = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.03, 900))), index=idx
    )
    perf = summarise(value, label="random")
    perf.verify()
    assert perf.cumulative_return_gap < 1e-12


def test_verify_raises_when_the_two_paths_disagree():
    """Guard the guard: a tampered check value must be caught."""
    perf = summarise(_curve(0.0003), label="x")
    perf.cumulative_return_check += 0.01
    with pytest.raises(AssertionError, match="disagrees between methods"):
        perf.verify()


def test_sharpe_daily_and_monthly_are_close():
    """Two sampling frequencies estimate the same quantity, so they should broadly agree.

    Not identical: monthly sampling discards within-month path and has far fewer
    observations. A large gap means something is wrong with the series, which is exactly
    why both are reported.
    """
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2021-07-29", periods=TRADING_DAYS * 4)
    value = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, len(idx)))), index=idx
    )
    perf = summarise(value, label="s")
    assert perf.sharpe_gap < 0.5, (perf.sharpe_daily, perf.sharpe_monthly)


def test_positive_drift_gives_positive_sharpe():
    perf = summarise(_curve(0.0006), label="up")
    assert perf.cumulative_return > 0


def test_risk_free_reduces_sharpe():
    """Raising the hurdle must lower the ratio. The handout's flat 2% got this wrong by
    understating the true rate over 2021-2026."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-07-29", periods=TRADING_DAYS * 3)
    value = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, len(idx)))), index=idx
    )
    low = summarise(value, pd.Series(0.001, index=idx), label="low_rf")
    high = summarise(value, pd.Series(0.05, index=idx), label="high_rf")
    assert high.sharpe_daily < low.sharpe_daily
    assert high.mean_rf > low.mean_rf


def test_handout_convention_differs_from_textbook_on_a_volatile_series():
    """Why both are reported. Mixing a geometric numerator with an arithmetic
    denominator is not a rounding difference at high volatility."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2021-07-29", periods=TRADING_DAYS * 4)
    # TSLA-like volatility.
    value = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.038, len(idx)))), index=idx
    )
    perf = summarise(value, pd.Series(0.04, index=idx), label="vol")
    assert perf.annual_vol > 0.45
    assert abs(perf.sharpe_daily - perf.sharpe_handout_convention) > 0.05


def test_max_drawdown_is_found_at_the_right_date():
    idx = pd.bdate_range("2023-01-02", periods=7)
    value = pd.Series([100.0, 120.0, 130.0, 91.0, 100.0, 110.0, 115.0], index=idx)
    perf = summarise(value, label="dd")
    assert perf.max_drawdown == pytest.approx(91.0 / 130.0 - 1.0)
    assert perf.max_drawdown_date == idx[3]


def test_downside_only_enters_sortino():
    """Sortino must exceed Sharpe when downside moves are smaller than upside ones."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(21)
    moves = rng.normal(0.001, 0.01, 400)
    moves[moves < 0] *= 0.4  # compress losses, leave gains alone
    value = pd.Series(100 * np.cumprod(1 + moves), index=idx)
    perf = summarise(value, label="skew")
    assert perf.sortino > perf.sharpe_daily


def test_calendar_years_compound_to_the_total():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=TRADING_DAYS * 3)
    value = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(idx)))), index=idx
    )
    perf = summarise(value, label="cal")
    compounded = (1.0 + perf.calendar_years).prod() - 1.0
    assert compounded == pytest.approx(perf.cumulative_return, rel=1e-9)


def test_daily_simple_rf_inverts_continuous_compounding():
    annual = pd.Series([0.05, 0.05])
    daily = daily_simple_rf(annual)
    # Compounding the daily rate over a year returns the continuous rate.
    assert (1 + daily.iloc[0]) ** TRADING_DAYS - 1 == pytest.approx(
        np.expm1(0.05), rel=1e-9
    )


def test_comparison_table_lines_strategies_up():
    a = summarise(_curve(0.0005), label="a")
    b = summarise(_curve(0.0002), label="b")
    table = comparison_table({"a": a, "b": b})
    assert list(table.columns) == ["a", "b"]
    assert table.loc["cumulative return", "a"] > table.loc["cumulative return", "b"]


@pytest.mark.parametrize("bad", [pd.Series(dtype=float), pd.Series([100.0, 101.0])])
def test_too_short_a_series_is_rejected(bad):
    with pytest.raises(ValueError, match="at least 3"):
        summarise(bad, label="short")


def test_nonpositive_value_is_rejected():
    """A curve that touches zero makes returns meaningless, so refuse rather than emit inf."""
    idx = pd.bdate_range("2023-01-02", periods=5)
    with pytest.raises(ValueError, match="positive"):
        summarise(pd.Series([100.0, 50.0, 0.0, 10.0, 20.0], index=idx), label="ruin")
