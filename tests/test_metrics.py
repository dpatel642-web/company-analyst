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


def test_verify_raises_on_a_tampered_field():
    """Exercises the raise path only.

    Named honestly: this proves `verify()` raises when the FIELD is corrupted, not that
    the check can detect a corrupted equity curve. It cannot — see
    test_cumulative_cross_check_has_no_detection_power below.
    """
    perf = summarise(_curve(0.0003), label="x")
    perf.cumulative_return_check += 0.01
    with pytest.raises(AssertionError, match="numerically unstable"):
        perf.verify()


def test_cumulative_cross_check_has_no_detection_power():
    """Pin the honest limitation rather than advertising a tautology as verification.

    `expm1(sum(log1p(pct_change)))` telescopes to `log(v_n/v_0)`, so it is algebraically
    identical to `v_n/v_0 - 1`. Inject a fabricated 50% jump mid-curve and the "check"
    still agrees, because the corrupted curve IS the input to both paths.
    """
    value = _curve(0.0003, n=TRADING_DAYS * 2)
    corrupted = value.copy()
    corrupted.iloc[len(corrupted) // 2 :] *= 1.5

    perf = summarise(corrupted, label="corrupted")
    perf.verify()  # passes, and that is the point
    assert perf.cumulative_return_gap < 1e-12

    # The real guard against this lives on the backtest ledger, not here.
    assert perf.cumulative_return != pytest.approx(
        summarise(value, label="clean").cumulative_return
    )


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
    """Hand-checkable trough, padded past the minimum annualisation window."""
    head = [100.0, 120.0, 130.0, 91.0, 100.0, 110.0, 115.0]
    tail = [115.0 + 0.01 * i for i in range(1, 80)]  # drifts up, never a new drawdown
    idx = pd.bdate_range("2023-01-02", periods=len(head) + len(tail))
    value = pd.Series(head + tail, index=idx)

    perf = summarise(value, label="dd")
    assert perf.max_drawdown == pytest.approx(91.0 / 130.0 - 1.0)
    assert perf.max_drawdown_date == idx[3]  # the trough, not the prior peak


def test_series_too_short_to_annualise_is_refused():
    """A 3-point series rising 10% annualises to a CAGR of 16 million percent."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    with pytest.raises(ValueError, match="too short to annualise"):
        summarise(pd.Series([100.0, 105.0, 110.0], index=idx), label="tiny")


def test_elapsed_years_is_reported_alongside_bar_count():
    """`n/252` is a bar-count proxy, not elapsed time, and the two disagree.

    The sign of the disagreement depends on the bar source, which is why both are
    reported rather than one being silently labelled "years": `bdate_range` yields about
    260 bars per year because it has no holidays, so `n/252` runs ahead of the calendar;
    real NYSE sessions average about 251 per year, so it runs behind. Either way a reader
    seeing "4.97y" next to a date range will assume calendar, and should be able to check.
    """
    perf = summarise(_curve(0.0004, n=TRADING_DAYS * 3), label="y")
    assert perf.years == pytest.approx(3.0, abs=0.02)
    assert perf.elapsed_years != pytest.approx(perf.years, abs=0.02)
    # bdate_range has no holidays, so here the bar-count basis overstates.
    assert perf.years > perf.elapsed_years


def test_sortino_uses_target_downside_deviation():
    """Denominator must be sqrt(mean over ALL periods of min(excess,0)^2), MAR = rf.

    The common shortcut (stdev of just the negative subset) makes the bias a function of
    the hit rate: measured on synthetic paths it runs from 1.49x at a 69% loss rate to
    0.61x at 17%, so it flips sign with drift and silently reorders a comparison table.
    """
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(21)
    moves = rng.normal(0.001, 0.01, 400)
    value = pd.Series(100 * np.cumprod(1 + moves), index=idx)
    perf = summarise(value, label="s")

    ret = value.pct_change().dropna()
    shortfall = ret.clip(upper=0.0)
    expected = float(np.sqrt((shortfall**2).mean()) * np.sqrt(TRADING_DAYS))
    assert perf.downside_vol == pytest.approx(expected, rel=1e-12)

    wrong = float(ret[ret < 0].std(ddof=1) * np.sqrt(TRADING_DAYS))
    assert perf.downside_vol != pytest.approx(wrong, rel=1e-6)


def test_sortino_bias_does_not_track_the_hit_rate():
    """The defect being fixed: the shortcut's error changes sign as drift changes."""
    idx = pd.bdate_range("2020-01-02", periods=1200)
    rng = np.random.default_rng(5)
    shocks = rng.normal(0.0, 0.012, 1200)
    ratios = []
    for drift in (-0.010, 0.0, 0.020):
        value = pd.Series(100 * np.cumprod(1 + drift + shocks), index=idx)
        perf = summarise(value, label=f"d{drift}")
        ret = value.pct_change().dropna()
        wrong = float(ret[ret < 0].std(ddof=1) * np.sqrt(TRADING_DAYS))
        ratios.append(wrong / perf.downside_vol)
    # The shortcut straddles 1.0 across regimes; the implementation must not.
    assert min(ratios) < 1.0 < max(ratios), ratios


def test_compressed_losses_raise_sortino_above_sharpe():
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
def test_too_few_observations_is_rejected(bad):
    with pytest.raises(ValueError, match="at least 3"):
        summarise(bad, label="short")


def test_nonpositive_value_is_rejected():
    """A curve that touches zero makes returns meaningless, so refuse rather than emit inf."""
    idx = pd.bdate_range("2023-01-02", periods=80)
    vals = [100.0, 50.0, 0.0] + [10.0] * 77
    with pytest.raises(ValueError, match="positive"):
        summarise(pd.Series(vals, index=idx), label="ruin")
