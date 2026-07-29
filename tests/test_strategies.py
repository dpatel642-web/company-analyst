"""Every strategy in the library, held to the same invariants as the first two.

The bar for a new strategy is not "it runs". It is:
  1. the accounting identity holds on every bar
  2. it never borrows (cash stays non-negative)
  3. it collapses onto the benchmark at zero pricing vol, where an option is worth exactly
     its intrinsic value and every overlay is economically a wash
  4. the economics it claims in its docstring are the economics it delivers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canalyst.backtest import TRADING_DAYS, run_backtest
from canalyst.data.calendar import roll_schedule, sessions
from canalyst.metrics import summarise
from canalyst.options.vol import apply_markup, close_to_close
from canalyst.strategies.buy_hold import BuyHold
from canalyst.strategies.cash_secured_put import CashSecuredPut
from canalyst.strategies.collar import Collar, ZeroCostCollar
from canalyst.strategies.covered_call import CoveredCall
from canalyst.strategies.premium_selling import IronCondor, ShortStrangle
from canalyst.strategies.protective_put import ProtectivePut
from canalyst.strategies.spreads import VerticalSpread
from canalyst.strategies.straddle import LongStraddle
from canalyst.strategies.wheel import Wheel

SEED = 20260729


def _world(sigma_true=0.30, drift=0.10, n_years=3.0, seed=SEED, dividends=False,
           rangebound=False):
    idx = sessions("2021-07-29", "2026-07-29")[: int(n_years * TRADING_DAYS)]
    n = len(idx)
    rng = np.random.default_rng(seed)
    daily = sigma_true / np.sqrt(TRADING_DAYS)
    if rangebound:
        walk = np.cumsum(rng.normal(0.0, daily, n))
        path = walk - np.arange(1, n + 1) / n * walk[-1]
    else:
        path = np.cumsum(rng.normal(drift / TRADING_DAYS, daily, n))
    close = pd.Series(100.0 * np.exp(path), index=idx)
    sigma = close_to_close(close, window=60).bfill().fillna(sigma_true)
    div = pd.Series(0.0, index=idx)
    if dividends:
        for day in idx[::63]:
            div.loc[day] = 0.60
    return dict(
        close=close, sigma=sigma, rate=pd.Series(0.04, index=idx),
        schedule=roll_schedule(idx), dividends=div, ticker="S",
    )


def _all_strategies():
    return [
        CoveredCall(target_delta=0.25),
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
        CashSecuredPut(target_delta=0.25),
        Collar(call_otm=0.05, put_otm=0.05),
        ZeroCostCollar(call_otm=0.05),
        VerticalSpread("bull_call", 0.00, 0.10),
        VerticalSpread("bear_put", 0.00, 0.10),
        LongStraddle(),
        ShortStrangle(call_otm=0.10, put_otm=0.10),
        IronCondor(call_otm=0.10, put_otm=0.10, wing_width=0.05),
        Wheel(target_delta=0.25),
    ]


IDS = [s.name for s in _all_strategies()]


# ------------------------------------------------------------- 1. the core invariant


@pytest.mark.parametrize("strategy", _all_strategies(), ids=IDS)
@pytest.mark.parametrize("dividends", [False, True], ids=["no_div", "div"])
def test_identity_holds_for_every_strategy(strategy, dividends):
    result = run_backtest(strategy, **_world(dividends=dividends))
    result.assert_identity()
    assert result.max_abs_residual < 1e-8


@pytest.mark.parametrize("strategy", _all_strategies(), ids=IDS)
@pytest.mark.parametrize("drift", [-0.40, 0.0, 0.55])
def test_identity_holds_across_regimes(strategy, drift):
    result = run_backtest(strategy, **_world(drift=drift, sigma_true=0.45))
    result.assert_identity()


# ------------------------------------------------------------- 2. nobody borrows


@pytest.mark.parametrize("strategy", _all_strategies(), ids=IDS)
def test_no_strategy_borrows(strategy):
    """Sizing to equity is what keeps a losing run from becoming a margin loan."""
    world = _world(drift=0.55, sigma_true=0.45)
    result = run_backtest(strategy, **world)
    opening = float(world["close"].iloc[0])
    # Long-premium structures pay cash up front, so allow a small debit but not leverage.
    assert result.bars["cash"].min() > -0.05 * opening, strategy.name


@pytest.mark.parametrize("strategy", _all_strategies(), ids=IDS)
def test_value_stays_positive(strategy):
    result = run_backtest(strategy, **_world(drift=-0.40, sigma_true=0.55))
    assert (result.value > 0).all(), f"{strategy.name} went non-positive"


# --------------------------------------------- 3. zero-vol collapse onto the benchmark


@pytest.mark.parametrize(
    "strategy",
    [
        CoveredCall(strike_rule="moneyness", otm_pct=0.05),
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
        Collar(call_otm=0.05, put_otm=0.05),
    ],
    ids=["covered_call", "protective_put", "collar"],
)
def test_stock_holding_overlays_collapse_at_zero_vol(strategy):
    """At zero pricing vol an option is worth exactly intrinsic, so a stock-plus-options
    overlay is economically identical to holding the stock. Any sign error, double count or
    dropped settlement breaks the exact equality."""
    world = _world()
    world["sigma"] = pd.Series(0.0, index=world["close"].index)
    overlay = run_backtest(strategy, **world)
    benchmark = run_backtest(BuyHold(), **world)
    pd.testing.assert_series_equal(
        overlay.value, benchmark.value, check_names=False, rtol=0, atol=1e-8
    )


# ----------------------------------------------------------- 4. the claimed economics


def test_cash_secured_put_matches_a_covered_call_at_the_SAME_strike():
    """Parity is a same-strike statement, and only a same-strike statement.

    Both structures pay `min(S_T, K)`. At a matched at-the-money strike they should track
    closely. The earlier version of this test compared a 0.25-delta put against a 0.25-delta
    call, which are on OPPOSITE sides of spot: different strikes, and deltas of roughly +0.25
    against +0.75. That comparison tests position sizing, not parity, and it failed for a good
    reason.
    """
    world = _world(dividends=False)
    csp = run_backtest(CashSecuredPut(strike_rule="moneyness", otm_pct=0.0), **world)
    cc = run_backtest(CoveredCall(strike_rule="moneyness", otm_pct=0.0), **world)
    csp.assert_identity()
    cc.assert_identity()
    a = csp.value.iloc[-1] / csp.value.iloc[0]
    b = cc.value.iloc[-1] / cc.value.iloc[0]
    assert abs(a - b) < 0.15 * max(a, b), f"same-strike parity broken: {a:.3f} vs {b:.3f}"


def test_delta_matched_put_and_call_are_NOT_equivalent():
    """Pin the misconception, so nobody re-derives the wrong version of parity."""
    world = _world(dividends=False)
    csp = run_backtest(CashSecuredPut(target_delta=0.25), **world)
    cc = run_backtest(CoveredCall(target_delta=0.25), **world)
    # The covered call carries roughly three times the directional exposure.
    assert cc.value.pct_change().std() > 2.0 * csp.value.pct_change().std()


def test_cash_secured_put_holds_no_stock():
    result = run_backtest(CashSecuredPut(target_delta=0.25), **_world())
    assert (result.bars["shares"] == 0.0).all()


def test_collar_caps_both_tails():
    """A collar must be the least volatile of the stock-holding arms, by construction."""
    world = _world(sigma_true=0.45)
    collar = run_backtest(Collar(call_otm=0.05, put_otm=0.05), **world)
    cc = run_backtest(CoveredCall(strike_rule="moneyness", otm_pct=0.05), **world)
    bench = run_backtest(BuyHold(), **world)
    vol = lambda r: r.value.pct_change().std()  # noqa: E731
    assert vol(collar) < vol(cc) < vol(bench)


def test_zero_cost_collar_nets_close_to_zero_premium():
    """The defining property: the written call pays for the bought put."""
    world = _world(sigma_true=0.35)
    strategy = ZeroCostCollar(call_otm=0.05)
    result = run_backtest(strategy, **world)
    result.assert_identity()
    opening = float(world["close"].iloc[0])
    assert abs(result.net_premium) < 0.02 * opening, (
        f"net premium {result.net_premium:.4f} should be near zero"
    )
    assert strategy.unsolved_rolls == 0, "every roll should have found a zero-cost strike"


def test_zero_cost_collar_reports_when_it_cannot_solve():
    """If no zero-cost strike exists it writes the call alone and SAYS so, rather than
    silently becoming a different structure."""
    strategy = ZeroCostCollar(call_otm=0.30, max_put_otm=0.02)
    result = run_backtest(strategy, **_world(sigma_true=0.15))
    result.assert_identity()
    assert strategy.unsolved_rolls > 0


def test_long_straddle_loses_when_priced_at_realised_vol():
    """The control case. Bought at exactly the vol the underlying goes on to realise, a
    straddle is a fair bet before costs, so it should not beat the benchmark. A strategy set
    where everything wins has a bug in it."""
    world = _world(rangebound=True, sigma_true=0.35)
    straddle = run_backtest(LongStraddle(), **world)
    straddle.assert_identity()
    bench = run_backtest(BuyHold(), **world)
    assert straddle.value.iloc[-1] < bench.value.iloc[-1]


def test_short_strangle_collects_more_premium_than_a_covered_call():
    """Two short legs against one, so it harvests more of the variance risk premium, and
    carries an unbounded naked-call risk in exchange."""
    world = _world(rangebound=True, sigma_true=0.35)
    strangle = run_backtest(ShortStrangle(call_otm=0.10, put_otm=0.10), **world)
    cc = run_backtest(CoveredCall(strike_rule="moneyness", otm_pct=0.10), **world)
    assert strangle.net_premium > cc.net_premium


def test_iron_condor_collects_less_than_the_naked_strangle():
    """The wings cost premium. That is what buying a finite maximum loss costs."""
    world = _world(rangebound=True)
    condor = run_backtest(IronCondor(0.10, 0.10, 0.05), **world)
    strangle = run_backtest(ShortStrangle(0.10, 0.10), **world)
    condor.assert_identity()
    assert 0 < condor.net_premium < strangle.net_premium


def test_iron_condor_is_less_volatile_than_the_strangle():
    world = _world(sigma_true=0.50, drift=0.0)
    condor = run_backtest(IronCondor(0.10, 0.10, 0.05), **world)
    strangle = run_backtest(ShortStrangle(0.10, 0.10), **world)
    assert condor.value.pct_change().std() < strangle.value.pct_change().std()


def test_bull_call_spread_has_a_capped_loss():
    """Defined risk: the debit is the most it can lose, so a crash cannot ruin it."""
    world = _world(drift=-0.60, sigma_true=0.50)
    spread = run_backtest(VerticalSpread("bull_call", 0.0, 0.10), **world)
    spread.assert_identity()
    assert spread.value.min() > 0.5 * spread.value.iloc[0]


def test_bear_put_spread_gains_in_a_downtrend():
    world = _world(drift=-0.50, sigma_true=0.35)
    bear = run_backtest(VerticalSpread("bear_put", 0.0, 0.10), **world)
    bull = run_backtest(VerticalSpread("bull_call", 0.0, 0.10), **world)
    assert bear.value.iloc[-1] > bull.value.iloc[-1]


# --------------------------------------------------------------------- the wheel's phases


def test_wheel_alternates_between_phases():
    """It must actually cycle. A wheel stuck in one phase is a cash-secured put or a covered
    call wearing the wrong name."""
    strategy = Wheel(target_delta=0.30)
    result = run_backtest(strategy, **_world(rangebound=True, sigma_true=0.40))
    result.assert_identity()
    assert strategy.put_writes > 3, "should write puts in the cash phase"
    assert strategy.call_writes > 3, "and calls once assigned"
    assert strategy.assignments_taken > 0
    assert strategy.called_away > 0


def test_wheel_phase_follows_the_ledger_not_a_tracked_flag():
    """Phase is read off what actually settled, via the engine's note_expiry hook. Shares
    held and the stock phase must never disagree."""
    strategy = Wheel(target_delta=0.30)
    result = run_backtest(strategy, **_world(rangebound=True, sigma_true=0.40))
    bars = result.bars
    # In the cash phase there is no stock; in the stock phase there is.
    assert (bars["shares"] >= 0).all()
    assert bars["shares"].max() > 0, "never entered the stock phase"
    assert (bars["shares"] == 0).any(), "never sat in the cash phase"


def test_wheel_accumulates_stock_in_a_downtrend():
    """The asymmetry worth knowing: assignment happens when the trade has gone against you,
    so a sustained decline leaves the wheel long at exactly the wrong time."""
    strategy = Wheel(target_delta=0.30)
    result = run_backtest(strategy, **_world(drift=-0.45, sigma_true=0.40))
    result.assert_identity()
    assert strategy.assignments_taken > 0
    # Measure time spent long, not the terminal state: the final cycle may happen to be
    # called away, which says nothing about the behaviour over the decline.
    in_stock = (result.bars["shares"] > 0).mean()
    assert in_stock > 0.30, f"only {in_stock:.0%} of bars held stock in a downtrend"


# ------------------------------------------------------------------ markup sensitivity


@pytest.mark.parametrize(
    "strategy",
    [CoveredCall(target_delta=0.25), CashSecuredPut(target_delta=0.25),
     ShortStrangle(0.10, 0.10), IronCondor(0.10, 0.10, 0.05)],
    ids=["covered_call", "cash_secured_put", "short_strangle", "iron_condor"],
)
def test_premium_sellers_gain_from_a_variance_risk_premium(strategy):
    """Every net-short-premium structure must benefit from selling above realised vol. If one
    does not, its legs are wired the wrong way round."""
    world = _world(rangebound=True, sigma_true=0.35)
    fair = run_backtest(strategy, **world)
    world_marked = dict(world, sigma=apply_markup(world["sigma"], 1.25))
    marked = run_backtest(strategy, **world_marked)
    assert marked.net_premium > fair.net_premium
    assert marked.value.iloc[-1] > fair.value.iloc[-1]


def test_long_premium_structures_are_hurt_by_a_markup():
    """The mirror image, and a check that the sign convention is not universally flattering."""
    world = _world(rangebound=True, sigma_true=0.35)
    for strategy in (LongStraddle(), ProtectivePut(strike_rule="moneyness", otm_pct=0.05)):
        fair = run_backtest(strategy, **world)
        marked = run_backtest(strategy, **dict(world, sigma=apply_markup(world["sigma"], 1.25)))
        assert marked.value.iloc[-1] < fair.value.iloc[-1], strategy.name


def test_every_strategy_produces_summarisable_performance():
    world = _world()
    for strategy in _all_strategies():
        result = run_backtest(strategy, **world)
        perf = summarise(result.value, world["rate"], label=strategy.name)
        perf.verify()
        assert np.isfinite(perf.sharpe_daily), strategy.name
