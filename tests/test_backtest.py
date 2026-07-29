"""Phase D gate: the engine's accounting must be airtight.

The load-bearing tests:

  test_*_identity_holds        every bar's value change is fully explained
  test_zero_vol_*_collapses    a zero-premium covered call IS buy-and-hold, exactly
  test_handout_bug_*           the two handout bugs are reproduced, then shown to
                               violate the invariant that the real engine satisfies

That last group matters most. It is not enough for the new engine to pass; the old
approach has to demonstrably fail the same check, or the check is not discriminating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canalyst.backtest import TRADING_DAYS, run_backtest
from canalyst.data.calendar import roll_schedule, sessions
from canalyst.options.bs import bs_delta, bs_price, strike_from_delta
from canalyst.options.vol import (
    apply_markup,
    close_to_close,
    trailing_dividend_yield,
)
from canalyst.strategies.buy_hold import BuyHold
from canalyst.strategies.covered_call import CoveredCall
from canalyst.strategies.protective_put import ProtectivePut

SEED = 20260729


def _world(
    n_years: float = 3.0,
    sigma_true: float = 0.55,
    drift: float = 0.10,
    rate: float = 0.04,
    start: str = "2021-07-29",
    seed: int = SEED,
    flat: bool = False,
    rangebound: bool = False,
):
    """A synthetic but structurally real world: NYSE sessions, GBM prices, monthly rolls.

    `rangebound` is a Brownian bridge, not a zero-drift walk and not an oscillation.

    Zero drift constrains the *expected* log return, not the realised one: this seed's
    zero-drift path runs 100 to 270 over three years, and a covered call rightly loses
    on it. A sine wave is worse in the opposite direction, because its up-legs are large
    and predictable, which is the single worst environment for a short call.

    A bridge is the right object. Increments stay essentially iid, so there is no
    exploitable within-month trend, while the path is pinned to finish where it started.
    """
    all_sessions = sessions(start, "2026-07-29")
    idx = all_sessions[: int(n_years * TRADING_DAYS)]
    n = len(idx)

    if flat:
        close = pd.Series(100.0, index=idx)
    elif rangebound:
        rng = np.random.default_rng(seed)
        daily_vol = sigma_true / np.sqrt(TRADING_DAYS)
        walk = np.cumsum(rng.normal(0.0, daily_vol, n))
        # Subtract the linear path to the terminal value: a Brownian bridge.
        bridge = walk - np.arange(1, n + 1) / n * walk[-1]
        close = pd.Series(100.0 * np.exp(bridge), index=idx)
    else:
        rng = np.random.default_rng(seed)
        daily_vol = sigma_true / np.sqrt(TRADING_DAYS)
        shocks = rng.normal(drift / TRADING_DAYS, daily_vol, n)
        close = pd.Series(100.0 * np.exp(np.cumsum(shocks)), index=idx)

    sigma = close_to_close(close, window=60).bfill().fillna(sigma_true)
    rates = pd.Series(rate, index=idx)
    schedule = roll_schedule(idx)
    return close, sigma, rates, schedule


def _run(strategy, flat=False, **kw):
    close, sigma, rates, schedule = _world(flat=flat, **kw)
    return run_backtest(strategy, close, sigma, rates, schedule, ticker="SYNTH")


# ------------------------------------------------------------------- the core invariant


@pytest.mark.parametrize(
    "strategy",
    [
        BuyHold(),
        CoveredCall(strike_rule="delta", target_delta=0.25),
        CoveredCall(strike_rule="moneyness", otm_pct=0.05),
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
    ],
    ids=["buy_hold", "cc_delta", "cc_moneyness", "protective_put"],
)
def test_identity_holds_every_bar(strategy):
    result = _run(strategy)
    result.assert_identity()
    assert result.max_abs_residual < 1e-9


@pytest.mark.parametrize("sigma_true", [0.15, 0.40, 0.90])
@pytest.mark.parametrize("drift", [-0.30, 0.0, 0.45])
def test_identity_holds_across_regimes(sigma_true, drift):
    """A leak that only appears in a crash or a melt-up is still a leak."""
    result = _run(
        CoveredCall(target_delta=0.25), sigma_true=sigma_true, drift=drift
    )
    result.assert_identity()


def test_identity_holds_with_dividends_and_fees():
    close, sigma, rates, schedule = _world()
    dividends = pd.Series(0.0, index=close.index)
    for day in close.index[::63]:  # a quarterly payer
        dividends.loc[day] = 0.85
    result = run_backtest(
        CoveredCall(target_delta=0.25),
        close, sigma, rates, schedule,
        dividends=dividends,
        fee_per_contract=0.65,
        ticker="SYNTH",
    )
    result.assert_identity()
    assert result.fees_paid > 0
    assert result.bars["dividends"].sum() == pytest.approx(
        (result.bars["shares"].shift(1).fillna(0.0) * dividends).sum()
    )


def test_residual_is_reported_not_swallowed():
    result = _run(CoveredCall(target_delta=0.25))
    assert "residual" in result.bars.columns
    assert result.bars["residual"].notna().all()


# ------------------------------------------------------- the collapse: overlay == benchmark


def test_zero_vol_covered_call_collapses_onto_buy_and_hold():
    """With no volatility a call is worth its intrinsic value and the overlay is a wash.

    This is the strongest single check that the option leg is wired up correctly: any
    sign error, double-count, or dropped settlement breaks the exact equality.
    """
    close, _, rates, schedule = _world()
    zero_vol = pd.Series(0.0, index=close.index)

    overlay = run_backtest(
        CoveredCall(strike_rule="moneyness", otm_pct=0.05),
        close, zero_vol, rates, schedule, ticker="SYNTH",
    )
    benchmark = run_backtest(BuyHold(), close, zero_vol, rates, schedule, ticker="SYNTH")

    pd.testing.assert_series_equal(
        overlay.value, benchmark.value, check_names=False, rtol=0, atol=1e-9
    )


def test_zero_vol_protective_put_also_collapses():
    close, _, rates, schedule = _world()
    zero_vol = pd.Series(0.0, index=close.index)
    overlay = run_backtest(
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
        close, zero_vol, rates, schedule, ticker="SYNTH",
    )
    benchmark = run_backtest(BuyHold(), close, zero_vol, rates, schedule, ticker="SYNTH")
    pd.testing.assert_series_equal(
        overlay.value, benchmark.value, check_names=False, rtol=0, atol=1e-9
    )


def test_buy_hold_tracks_the_price_exactly():
    close, sigma, rates, schedule = _world()
    result = run_backtest(BuyHold(), close, sigma, rates, schedule, ticker="SYNTH")
    # Funded with one share's price, holding one share, no options: value IS the price.
    pd.testing.assert_series_equal(
        result.value, close, check_names=False, rtol=0, atol=1e-9
    )


# ------------------------------------------------------------- economics that must hold


def test_trendless_path_premise_holds():
    """Guard the bridge itself: volatile day to day, going nowhere overall."""
    close, _, _, _ = _world(rangebound=True)
    assert abs(close.iloc[-1] / close.iloc[0] - 1.0) < 0.05
    assert close.pct_change().std() * np.sqrt(TRADING_DAYS) > 0.30


def test_covered_call_needs_the_variance_risk_premium_to_win():
    """The strategy's edge IS implied trading above realised. Priced at realised vol,
    writing a call is close to a fair bet and cannot be expected to beat the benchmark.

    This is the most important economic fact in the whole project, because the backtest
    has no historical implied vol to price from. Marking premiums up to where implied
    actually trades turns a coin flip into an edge, on the same price path.
    """
    close, realised, rates, schedule = _world(rangebound=True)
    benchmark = run_backtest(BuyHold(), close, realised, rates, schedule, ticker="SYNTH")

    strategy = CoveredCall(target_delta=0.25)
    # Same price path, same strategy. The only difference is the vol options trade at.
    fair = run_backtest(
        strategy, close, realised, rates, schedule, ticker="SYNTH"
    )
    with_vrp = run_backtest(
        strategy, close, apply_markup(realised, 1.20), rates, schedule, ticker="SYNTH"
    )

    final = lambda r: r.value.iloc[-1]  # noqa: E731
    assert with_vrp.net_premium > fair.net_premium
    assert final(with_vrp) > final(fair), "selling above realised vol must pay"
    assert final(with_vrp) > final(benchmark), "with a real VRP the overlay wins"


def test_covered_call_loses_return_in_a_melt_up():
    """The honest cost. An overlay that wins in every regime would be a bug."""
    close, sigma, rates, schedule = _world(sigma_true=0.35, drift=0.80)
    overlay = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule, ticker="SYNTH"
    )
    benchmark = run_backtest(BuyHold(), close, sigma, rates, schedule, ticker="SYNTH")
    assert overlay.value.iloc[-1] < benchmark.value.iloc[-1]


def test_covered_call_cuts_volatility():
    close, sigma, rates, schedule = _world(sigma_true=0.55, drift=0.10)
    overlay = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule, ticker="SYNTH"
    )
    benchmark = run_backtest(BuyHold(), close, sigma, rates, schedule, ticker="SYNTH")
    assert overlay.value.pct_change().std() < benchmark.value.pct_change().std()


def test_premium_is_actually_collected():
    result = _run(CoveredCall(target_delta=0.25))
    assert result.rolls > 20
    # Premium received, not terminal cash: on an upward path the assignments paid out
    # can exceed the premium taken in, so terminal cash says nothing about collection.
    assert result.net_premium > 0


def test_short_call_marks_as_a_liability():
    result = _run(CoveredCall(target_delta=0.25))
    live = result.bars[result.bars["open_positions"] > 0]
    assert len(live) > 0
    assert (live["option_mtm"] <= 1e-9).all(), "a written call is never an asset"


def test_only_one_call_is_ever_open():
    """Stacking a second short call would be uncovered, and the guard must hold."""
    result = _run(CoveredCall(target_delta=0.25))
    assert result.bars["open_positions"].max() == 1


def test_higher_delta_collects_more_premium():
    close, sigma, rates, schedule = _world(drift=0.0)
    cheap = run_backtest(
        CoveredCall(target_delta=0.10), close, sigma, rates, schedule, ticker="S"
    )
    rich = run_backtest(
        CoveredCall(target_delta=0.40), close, sigma, rates, schedule, ticker="S"
    )
    assert rich.net_premium > cheap.net_premium


def test_pricing_vol_markup_increases_premium_collected():
    """The markup belongs to the vol series, not the strategy: it is what the market
    quotes, so it must govern the opening price and every later mark alike."""
    close, sigma, rates, schedule = _world(drift=0.0)
    base = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule, ticker="S"
    )
    marked = run_backtest(
        CoveredCall(target_delta=0.25), close, apply_markup(sigma, 1.25), rates,
        schedule, ticker="S",
    )
    assert marked.net_premium > base.net_premium


def test_interest_accrues_on_idle_cash():
    """Idle cash must earn the prevailing rate.

    Requires `fully_invested=False`: the default benchmark sweeps cash into stock on
    every bar, so by construction it holds no idle cash to earn anything. That sweep is
    deliberate (it is what makes the benchmark comparable to the collateralised
    overlays), which means interest has to be tested on a strategy that does hold cash.
    """
    close, sigma, rates, schedule = _world(drift=0.0)
    idle = float(close.iloc[0]) * 3.0  # funds one share and leaves 2x sitting in cash
    kw = dict(
        close=close, sigma=sigma, schedule=schedule,
        starting_equity=idle, ticker="S",
    )
    at_zero = run_backtest(
        BuyHold(fully_invested=False), rate=pd.Series(0.0, index=close.index), **kw
    )
    at_five = run_backtest(
        BuyHold(fully_invested=False), rate=pd.Series(0.05, index=close.index), **kw
    )
    assert at_five.bars["interest"].sum() > at_zero.bars["interest"].sum()
    assert at_zero.bars["interest"].sum() == pytest.approx(0.0, abs=1e-12)


def test_fully_invested_benchmark_holds_no_idle_cash():
    """The fix for the dividend-policy mismatch: the benchmark must sweep cash to stock.

    A constant-one-share benchmark leaves dividends idle while the collateralised
    overlays compound theirs into stock, so the comparison silently measures dividend
    policy. Measured on a PG-like path that mismatch was worth 7.26 percentage points,
    all of it flattering the overlay.
    """
    close, sigma, rates, schedule = _world()
    dividends = pd.Series(0.0, index=close.index)
    for day in close.index[::63]:
        dividends.loc[day] = 0.85

    swept = run_backtest(
        BuyHold(), close, sigma, rates, schedule, dividends=dividends, ticker="S"
    )
    idle = run_backtest(
        BuyHold(fully_invested=False), close, sigma, rates, schedule,
        dividends=dividends, ticker="S",
    )
    swept.assert_identity()
    idle.assert_identity()

    assert abs(swept.bars["cash"].iloc[-1]) < 1e-9, "fully invested means no idle cash"
    assert idle.bars["cash"].iloc[-1] > 1.0, "the old convention parks dividends in cash"
    assert swept.bars["shares"].iloc[-1] > 1.0, "dividends must compound into stock"
    assert idle.bars["shares"].iloc[-1] == pytest.approx(1.0)
    # And the convention alone changes the benchmark, which is the whole problem.
    assert swept.value.iloc[-1] != pytest.approx(idle.value.iloc[-1], rel=1e-6)


def test_zero_vol_collapse_still_holds_with_dividends():
    """The collapse test that did not exist, and that the mismatch was breaking.

    At zero pricing vol a written call is worth exactly intrinsic, so a covered call is
    economically identical to buy-and-hold. Without matched dividend policy this broke by
    12.3% of initial capital in the overlay's favour while the accounting identity passed,
    which meant the test was measuring dividend policy rather than options.
    """
    close, _, rates, schedule = _world()
    zero_vol = pd.Series(0.0, index=close.index)
    dividends = pd.Series(0.0, index=close.index)
    for day in close.index[::63]:
        dividends.loc[day] = 0.85

    kw = dict(
        close=close, sigma=zero_vol, rate=rates, schedule=schedule,
        dividends=dividends, ticker="S",
    )
    overlay = run_backtest(CoveredCall(strike_rule="moneyness", otm_pct=0.05), **kw)
    benchmark = run_backtest(BuyHold(), **kw)
    pd.testing.assert_series_equal(
        overlay.value, benchmark.value, check_names=False, rtol=0, atol=1e-8
    )


# ---------------------------------------------------- the handout's bugs, reproduced


def test_handout_bug_a_omitting_interim_mtm_violates_the_identity():
    """Reproduce Bug A: value = shares*close + cash, with no interim option mark.

    Cash moves only on roll and expiry days, so between them the short call's liability
    is invisible. The reconstructed series must fail the very check the real engine
    passes, which is what makes the check meaningful.
    """
    result = _run(CoveredCall(strike_rule="moneyness", otm_pct=0.05))
    bars = result.bars

    # The handout's notion of value: share leg plus cash, option liability ignored.
    handout_value = bars["shares"] * bars["spot"] + bars["cash"]
    explained = (
        bars["share_pnl"] + bars["dividends"] + bars["interest"]
        + bars["option_mtm_change"] - bars["fees"]
    )
    handout_residual = handout_value.diff() - explained

    assert handout_residual.abs().max() > 1.0, (
        "Bug A should leave a large unexplained gap"
    )
    # And the real engine's own residual is essentially zero on the same path.
    assert result.max_abs_residual < 1e-9


def test_handout_bug_a_distorts_volatility_not_just_level():
    """Why Bug A corrupts Sharpe specifically.

    Ignoring the interim mark suppresses the option's day-to-day contribution and then
    dumps it in one lump at expiry. The daily return series is therefore wrong even
    where the terminal value is nearly right.
    """
    result = _run(CoveredCall(strike_rule="moneyness", otm_pct=0.05))
    bars = result.bars
    handout_value = bars["shares"] * bars["spot"] + bars["cash"]

    true_vol = result.value.pct_change().std()
    handout_vol = handout_value.pct_change().std()
    assert not np.isclose(true_vol, handout_vol, rtol=0.01), (
        f"vol should differ materially: {true_vol:.6f} vs {handout_vol:.6f}"
    )

    # The lump arrives on expiry days: the largest absolute daily changes in the
    # handout series cluster on settlement dates.
    expiries = set(_world()[3]["expiry"])
    worst_days = handout_value.diff().abs().nlargest(10).index
    assert sum(d in expiries for d in worst_days) >= 3


def test_handout_bug_b_dropped_expiry_payoff_loses_money_silently():
    """Reproduce Bug B: on a day that is both expiry and roll, the expiring position's
    payoff is overwritten by the new position's mark instead of being banked.

    On an upward-drifting path the dropped payoffs are mostly short-call losses, so the
    corrupted curve is biased *upward* relative to the truth. A backtest that looks
    better than reality is the dangerous direction.
    """
    close, sigma, rates, schedule = _world(sigma_true=0.45, drift=0.35)
    strategy = CoveredCall(strike_rule="moneyness", otm_pct=0.05)
    truth = run_backtest(strategy, close, sigma, rates, schedule, ticker="S")

    # Rebuild the ledger while discarding each settlement cash flow.
    bars = truth.bars
    settlement = pd.Series(0.0, index=bars.index)
    for day in schedule["expiry"]:
        if day in bars.index:
            i = bars.index.get_loc(day)
            # The mark the expiring option carried into settlement.
            settlement.iloc[i] = bars["option_mtm_change"].iloc[i]

    leaked = truth.value - settlement.cumsum()
    assert leaked.iloc[-1] != pytest.approx(truth.value.iloc[-1], rel=1e-6), (
        "dropping settlements must change the answer"
    )
    assert truth.max_abs_residual < 1e-9


# ------------------------------------------------------------------------- input guards


def test_mismatched_index_is_rejected():
    close, sigma, rates, schedule = _world()
    with pytest.raises(ValueError, match="sigma index"):
        run_backtest(BuyHold(), close, sigma.iloc[:-5], rates, schedule)
    with pytest.raises(ValueError, match="rate index"):
        run_backtest(BuyHold(), close, sigma, rates.iloc[:-5], schedule)


def test_empty_series_is_rejected():
    close, sigma, rates, schedule = _world()
    empty = close.iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        run_backtest(BuyHold(), empty, empty, empty, schedule)


def test_uncovered_write_is_refused():
    with pytest.raises(ValueError, match="not covered"):
        CoveredCall(shares=1.0, contracts=2.0)


def test_nonsense_strategy_parameters_are_refused():
    with pytest.raises(ValueError):
        CoveredCall(strike_rule="vibes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CoveredCall(target_delta=0.0)
    with pytest.raises(ValueError):
        ProtectivePut(otm_pct=1.5)


def test_strategy_cannot_see_the_future():
    """The context handed to a strategy exposes today and the known schedule only."""
    seen: list = []

    class Spy(CoveredCall):
        def options_to_open(self, ctx):
            seen.append(ctx)
            return super().options_to_open(ctx)

    close, sigma, rates, schedule = _world()
    run_backtest(Spy(target_delta=0.25), close, sigma, rates, schedule, ticker="S")
    assert seen
    for ctx in seen:
        assert set(vars(ctx)) == {
            "date", "spot", "sigma", "rate", "dividend",
            "is_roll", "expiry", "years_to_expiry", "open_positions", "equity",
            "div_yield",
        }
        if ctx.expiry is not None:
            assert ctx.expiry > ctx.date


# -------------------------------------------------------- settlement is priced correctly


def test_expiring_call_settles_at_intrinsic():
    """Hand-checkable: one roll, one expiry, known strike, known terminal price."""
    idx = sessions("2025-01-02", "2025-04-30")
    close = pd.Series(100.0, index=idx)
    # Push the price above the strike just before expiry so the call finishes ITM.
    schedule = roll_schedule(idx)
    roll_day, expiry_day = schedule.index[0], schedule["expiry"].iloc[0]
    close.loc[close.index > roll_day] = 130.0

    sigma = pd.Series(0.30, index=idx)
    rates = pd.Series(0.04, index=idx)
    strategy = CoveredCall(strike_rule="moneyness", otm_pct=0.05)
    result = run_backtest(
        strategy, close, sigma, rates, schedule.iloc[:1], ticker="S"
    )
    result.assert_identity()

    strike = round(100.0 * 1.05, 2)
    # q must match what the engine used. This test previously defaulted q to 0 and so
    # pinned the wrong convention: fixing the engine would have broken it. No dividends
    # here, so q is 0 for a real reason rather than by omission.
    assert result.bars["dividends"].sum() == 0.0
    premium = bs_price(
        100.0,
        strike,
        (idx.get_loc(expiry_day) - idx.get_loc(roll_day)) / TRADING_DAYS,
        0.04,
        0.30,
        "call",
        q=0.0,
    )
    settled = result.bars.loc[expiry_day]
    assert settled["open_positions"] == 0
    assert result.assignments == 1

    # Check value, not cash. Once the book empties the strategy rebalances shares to
    # equity/spot, which sweeps cash to zero and expresses the loss as a smaller
    # position instead. Value is the quantity that has to be right either way:
    #   one share at 130, plus the premium taken in, less the 130-against-105 intrinsic
    #   paid away, plus interest earned on cash along the way.
    interest = result.bars["interest"].cumsum().loc[expiry_day]
    expected_value = 130.0 + premium - (130.0 - strike) + interest
    assert settled["value"] == pytest.approx(expected_value, abs=1e-6)
    assert settled["cash"] == pytest.approx(0.0, abs=1e-6)
    assert settled["shares"] == pytest.approx(expected_value / 130.0, abs=1e-9)


# ------------------------------------------- collateralisation: never lever on a loss

# Note on what is measured here. share_value/equity is NOT a leverage metric for an
# option position: a short call caps equity while the share notional keeps rising, so
# that ratio exceeds 1 for any healthy covered call after a rally. Borrowing is what
# matters, so the guard is on cash, on the covered condition, and on the volatility
# property that the uncollateralised version actually violated.

NEG_CASH_TOL = -1e-6


@pytest.mark.parametrize("drift", [-0.45, 0.0, 0.60])
@pytest.mark.parametrize("sigma_true", [0.35, 0.75])
def test_covered_call_never_borrows(drift, sigma_true):
    """Assignment losses must shrink the position, not open a margin loan.

    Measured on real TSLA data, the fixed-one-share version drove cash to -240 against a
    222 share, so 4.4x the remaining equity sat in stock funded by credit. Volatility then
    printed 93.7% against the stock's 59.7%, which is impossible for a real covered call.
    """
    result = _run(CoveredCall(target_delta=0.25), drift=drift, sigma_true=sigma_true)
    result.assert_identity()
    assert result.bars["cash"].min() >= NEG_CASH_TOL


def test_written_calls_are_always_covered_by_shares():
    result = _run(CoveredCall(target_delta=0.25), drift=0.60, sigma_true=0.75)
    live = result.bars[result.bars["open_positions"] > 0]
    assert len(live) > 0
    assert (live["shares"] > 0).all()


def test_covered_call_deleverages_after_a_loss():
    """The mechanism itself: shares must fall once assignments have eaten capital."""
    result = _run(CoveredCall(target_delta=0.25), drift=0.60, sigma_true=0.75)
    assert result.assignments > 0
    assert result.bars["shares"].min() < 1.0


@pytest.mark.parametrize("drift", [-0.45, 0.0, 0.60])
def test_covered_call_volatility_stays_below_the_underlying(drift):
    """Long stock plus a short call has net delta under one, so its volatility must be
    under the underlying's. The uncollateralised version violated this on TSLA."""
    close, sigma, rates, schedule = _world(drift=drift, sigma_true=0.65)
    overlay = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule, ticker="S"
    )
    benchmark = run_backtest(BuyHold(), close, sigma, rates, schedule, ticker="S")
    assert overlay.value.pct_change().std() < benchmark.value.pct_change().std()


def test_uncollateralised_mode_borrows_persistently():
    """Keep the old behaviour reachable, and prove it is the thing that misbehaves.

    The accounting was never wrong in either mode; only the economic model was.
    """
    kw = dict(drift=0.60, sigma_true=0.75)
    good = _run(CoveredCall(target_delta=0.25), **kw)
    bad = _run(CoveredCall(target_delta=0.25, fully_collateralised=False), **kw)
    good.assert_identity()
    bad.assert_identity()

    bars = len(bad.bars)
    borrowed = int((bad.bars["cash"] < NEG_CASH_TOL).sum())
    assert borrowed > bars * 0.25, "the old model should borrow on many bars"
    assert int((good.bars["cash"] < NEG_CASH_TOL).sum()) == 0
    assert bad.bars["shares"].nunique() == 1, "the old model never deleverages"


def test_collateralised_and_uncollateralised_agree_before_any_assignment():
    """Until the first assignment there is nothing to deleverage, so they must match."""
    close, sigma, rates, schedule = _world()
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule, ticker="S")
    on = run_backtest(CoveredCall(target_delta=0.25), **kw)
    off = run_backtest(
        CoveredCall(target_delta=0.25, fully_collateralised=False), **kw
    )
    first_roll = schedule.index[0]
    upto = on.bars.index <= first_roll
    pd.testing.assert_series_equal(
        on.value[upto], off.value[upto], check_names=False, rtol=0, atol=1e-9
    )


# --------------------------------------- dividend yield reaches the pricer (BACKLOG #1)


def _dividend_world(annual_yield: float = 0.024, **kw):
    """A world with a real quarterly dividend, for exercising q."""
    close, sigma, rates, schedule = _world(**kw)
    dividends = pd.Series(0.0, index=close.index)
    per_quarter = float(close.iloc[0]) * annual_yield / 4.0
    for day in close.index[::63]:
        dividends.loc[day] = per_quarter
    return close, sigma, rates, schedule, dividends


def test_dividend_yield_is_derived_when_not_supplied():
    """Omitting q must not silently mean zero on a payer."""
    close, sigma, rates, schedule, dividends = _dividend_world()
    q = trailing_dividend_yield(dividends, close)
    assert q.max() > 0.015, "a 2.4% payer should register a positive yield"
    assert (q >= 0).all()


def test_zero_dividend_gives_zero_yield():
    close, sigma, rates, schedule = _world()
    q = trailing_dividend_yield(pd.Series(0.0, index=close.index), close)
    assert (q == 0.0).all()


def test_dividend_yield_lowers_the_call_premium_collected():
    """The defect: q=0 prices a call as if no dividend were coming, so the writer is paid
    twice. Supplying the true q must reduce the premium booked."""
    close, sigma, rates, schedule, dividends = _dividend_world()
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule,
              dividends=dividends, ticker="S")
    with_q = run_backtest(CoveredCall(target_delta=0.25), **kw)
    without_q = run_backtest(
        CoveredCall(target_delta=0.25),
        div_yield=pd.Series(0.0, index=close.index), **kw
    )
    with_q.assert_identity()
    without_q.assert_identity()
    assert with_q.net_premium < without_q.net_premium, (
        "pricing with q must collect less than pricing as if no dividend existed"
    )


def test_dividend_yield_shifts_the_chosen_strike():
    """q enters strike selection too. With it omitted, a nominal 0.25-delta strike sits at
    0.2375, so the pre-specified parameter quietly stops being the one in force."""
    S, T, r, sigma, target = 160.0, 21 / 252, 0.042, 0.18, 0.25
    k_no_q = strike_from_delta(S, T, r, sigma, target, "call", q=0.0)
    k_with_q = strike_from_delta(S, T, r, sigma, target, "call", q=0.024)
    assert k_with_q < k_no_q
    # The q=0 strike, evaluated at the true q, is not the delta that was asked for.
    assert abs(bs_delta(S, k_no_q, T, r, sigma, "call", q=0.024)) < target - 0.005
    # The q-aware strike is.
    assert abs(bs_delta(S, k_with_q, T, r, sigma, "call", q=0.024)) == pytest.approx(
        target, abs=1e-9
    )


def test_identity_holds_with_a_dividend_yield():
    close, sigma, rates, schedule, dividends = _dividend_world()
    for strategy in (
        BuyHold(),
        CoveredCall(target_delta=0.25),
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
    ):
        result = run_backtest(
            strategy, close, sigma, rates, schedule, dividends=dividends, ticker="S"
        )
        result.assert_identity()


def test_dividend_yield_is_capped_so_strike_inversion_stays_solvable():
    """A special dividend or a collapsed price can imply a yield that pushes
    target*exp(qT) past 1, which has no solution. The cap keeps it invertible."""
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = pd.Series(10.0, index=idx)          # a price that has collapsed
    dividends = pd.Series(0.0, index=idx)
    dividends.iloc[10] = 8.0                    # an enormous special dividend
    q = trailing_dividend_yield(dividends, close, max_yield=0.25)
    assert q.max() <= np.log1p(0.25) + 1e-12
    # And it is still usable for strike selection.
    strike_from_delta(10.0, 21 / 252, 0.04, 0.30, 0.25, "call", q=float(q.max()))


# ============ the interim mark check: what the telescoping identity cannot see ============
#
# The identity is a closure check on the ledger. Over a position's life
# sum(mark_t - mark_{t-1}) collapses to mark_settle - mark_open, so every interim mark cancels
# out of it. An adversarial review built four mutants touching ONLY the interim marking line
# and ran them against the 27 load-bearing assertions in this file. An interim mark carrying
# zero theta failed 0 of 27, while moving Sharpe by 0.024 and annualised vol by 1.3pp.
#
# These tests reproduce that and show the lattice check catches what the identity structurally
# cannot.


def _mutant_mark(variant: str):
    """Return a replacement for backtest._mark that corrupts ONLY the interim time to expiry."""
    from canalyst.options.bs import bs_price

    def frozen(kind, strike, spot, years, rate, sigma, quantity, div_yield=0.0):
        # No time decay at all: the option is marked as if it never aged.
        return quantity * bs_price(
            spot, strike, max(21 / 252, 0.0), rate, max(sigma, 0.0), kind, q=div_yield
        )

    def wrong_daycount(kind, strike, spot, years, rate, sigma, quantity, div_yield=0.0):
        # 365 instead of 252: T is off by a factor of 0.69, so price is off by ~17%.
        return quantity * bs_price(
            spot, strike, max(years * 252 / 365, 0.0), rate, max(sigma, 0.0), kind,
            q=div_yield,
        )

    return {"frozen": frozen, "wrong_daycount": wrong_daycount}[variant]


@pytest.mark.parametrize("variant", ["frozen", "wrong_daycount"])
def test_lattice_check_catches_a_corrupt_interim_mark(monkeypatch, variant):
    """The load-bearing test. A wrong interim mark must be caught by SOMETHING."""
    import canalyst.backtest as engine

    close, sigma, rates, schedule = _world()
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule, ticker="S",
              verify_marks=16)

    monkeypatch.setattr(engine, "_mark", _mutant_mark(variant))
    corrupt = run_backtest(CoveredCall(target_delta=0.25), **kw)

    # The identity still passes: the corruption telescopes out of it entirely.
    corrupt.assert_identity()
    # The lattice check does not.
    with pytest.raises(AssertionError, match="disagrees with an independent lattice"):
        corrupt.assert_marks()
    assert corrupt.worst_mark_error > 0.05, corrupt.worst_mark_error


@pytest.mark.parametrize("variant", ["frozen", "wrong_daycount"])
def test_the_identity_alone_cannot_see_a_corrupt_interim_mark(monkeypatch, variant):
    """Pin the gap itself, so nobody concludes the identity is sufficient.

    Scope note, because it matters for what this proves. Patching `_mark` corrupts every
    pricing call, including the opening and the settlement, so terminal value moves here too.
    The adversarial review's mutants were surgical, touching only the interim marking line, and
    those left terminal value BIT-IDENTICAL while shifting Sharpe by 0.024 and annualised
    volatility by 1.3pp. Reproducing that surgically would need a test-only seam in production
    code, which is a worse trade than stating the scope.

    Either way the load-bearing claim holds and is what this asserts: the residual stays at
    floating-point noise, so the accounting identity certifies a run whose marks are wrong.
    Booking at price `p` and marking at the same `p` nets to zero for any `p`.
    """
    import canalyst.backtest as engine

    close, sigma, rates, schedule = _world()
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule, ticker="S")

    truth = run_backtest(CoveredCall(target_delta=0.25), **kw)
    monkeypatch.setattr(engine, "_mark", _mutant_mark(variant))
    corrupt = run_backtest(CoveredCall(target_delta=0.25), **kw)

    # The whole point: the identity passes on a run that is priced wrong throughout.
    assert corrupt.max_abs_residual < 1e-9, "the identity is blind to this"
    truth.assert_identity()
    corrupt.assert_identity()

    # And every risk statistic has moved, which is the damage the identity cannot report.
    truth_vol = truth.value.pct_change().std()
    corrupt_vol = corrupt.value.pct_change().std()
    assert not np.isclose(truth_vol, corrupt_vol, rtol=1e-3), (
        f"volatility should differ: {truth_vol:.6f} vs {corrupt_vol:.6f}"
    )


def test_correct_marks_pass_the_lattice_check():
    """The check must not fire on correct pricing, or it is noise."""
    close, sigma, rates, schedule = _world()
    result = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule,
        ticker="S", verify_marks=16,
    )
    result.assert_identity()
    result.assert_marks()
    assert len(result.mark_checks) > 5, "the sampler must actually sample something"
    assert result.worst_mark_error < 0.01, result.worst_mark_error


def test_mark_check_samples_only_interim_bars():
    """Open and expiry bars are already pinned by the identity, so checking them proves
    nothing about the marks in between."""
    close, sigma, rates, schedule = _world()
    result = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule,
        ticker="S", verify_marks=20,
    )
    assert result.mark_checks
    for check in result.mark_checks:
        assert check["years"] > 0.0, "an expiry bar carries no interim information"


def test_mark_check_is_off_by_default():
    """It costs roughly ten times a plain run, and a check that taxes every call gets
    switched off. Every real analysis path enables it explicitly instead."""
    close, sigma, rates, schedule = _world()
    result = run_backtest(CoveredCall(target_delta=0.25), close, sigma, rates, schedule)
    assert result.mark_checks == []
    result.assert_marks()  # a no-op rather than a false pass
    assert result.worst_mark_error == 0.0


def test_lattice_check_also_catches_a_missing_dividend_yield(monkeypatch):
    """It pins the mark LEVEL, so it catches any wrong pricing argument, not just time."""
    import canalyst.backtest as engine
    from canalyst.options.bs import bs_price

    def ignores_q(kind, strike, spot, years, rate, sigma, quantity, div_yield=0.0):
        return quantity * bs_price(spot, strike, years, rate, sigma, kind, q=0.0)

    close, sigma, rates, schedule, dividends = _dividend_world(annual_yield=0.06)
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule,
              dividends=dividends, ticker="S", verify_marks=16)

    monkeypatch.setattr(engine, "_mark", ignores_q)
    corrupt = run_backtest(CoveredCall(target_delta=0.25), **kw)
    corrupt.assert_identity()
    with pytest.raises(AssertionError, match="disagrees with an independent lattice"):
        corrupt.assert_marks()


def test_mark_check_does_not_false_positive_on_a_cheap_option():
    """Materiality matters, or the check becomes noise people switch off.

    Ordinary lattice discretisation error is a large FRACTION of a cheap out-of-the-money
    premium while being economically irrelevant. Measured on real WMT data the protective put
    reached 0.938% of premium against a 1% tolerance, on nothing but discretisation. A failure
    now needs the error to be material against SPOT as well.
    """
    close, sigma, rates, schedule = _world(sigma_true=0.20)
    result = run_backtest(
        ProtectivePut(strike_rule="moneyness", otm_pct=0.15),  # cheap, far OTM
        close, sigma, rates, schedule, ticker="S", verify_marks=20,
    )
    result.assert_identity()
    result.assert_marks()
    assert result.mark_checks
    # Some checks may be a large fraction of premium, yet none material against spot.
    assert max(c["spot_error"] for c in result.mark_checks) < 2e-4


def test_mark_check_still_catches_a_material_error_on_a_cheap_option(monkeypatch):
    """The materiality gate must not become a way of ignoring real errors."""
    import canalyst.backtest as engine

    close, sigma, rates, schedule = _world(sigma_true=0.20)
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule, ticker="S",
              verify_marks=20)
    monkeypatch.setattr(engine, "_mark", _mutant_mark("frozen"))
    corrupt = run_backtest(CoveredCall(target_delta=0.25), **kw)
    corrupt.assert_identity()
    with pytest.raises(AssertionError, match="disagrees with an independent lattice"):
        corrupt.assert_marks()


# ================== the last three round-one findings, each pinned by a repro ==================


def test_fees_scale_with_contract_count():
    """Charged per SPEC, one fee per leg, understated costs by the position size.

    At 100 contracts over 35 rolls it billed 22.75 instead of 2275.00, a 100x error invisible
    at unit size and appearing the moment anyone scales the notional.
    """
    close, sigma, rates, schedule = _world()
    kw = dict(close=close, sigma=sigma, rate=rates, schedule=schedule, ticker="S",
              fee_per_contract=0.65)

    one = run_backtest(
        CoveredCall(shares=1.0, contracts=1.0, fully_collateralised=False), **kw
    )
    hundred = run_backtest(
        CoveredCall(shares=100.0, contracts=100.0, fully_collateralised=False),
        starting_equity=float(close.iloc[0]) * 100.0, **kw
    )
    one.assert_identity()
    hundred.assert_identity()
    assert hundred.fees_paid == pytest.approx(100.0 * one.fees_paid, rel=1e-9)
    assert hundred.fees_paid == pytest.approx(0.65 * 100.0 * hundred.rolls, rel=1e-9)


@pytest.mark.parametrize("series_name", ["close", "sigma", "rate"])
def test_nan_inputs_are_rejected_at_the_door(series_name):
    """A NaN does not fail loudly, it DISABLES the guards: every downstream check is a
    comparison, and comparisons against NaN are False."""
    close, sigma, rates, schedule = _world()
    args = {"close": close.copy(), "sigma": sigma.copy(), "rate": rates.copy()}
    args[series_name].iloc[400] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(CoveredCall(target_delta=0.25), schedule=schedule, ticker="S", **args)


def test_assert_identity_rejects_a_nan_residual():
    """pandas .max() is skipna=True, so `nan > tol` is False and an all-NaN residual PASSED.

    One NaN in the rate series propagated through interest into cash, `summarise`'s dropna()
    then amputated 17 months, and both the identity and verify() certified the result.
    """
    close, sigma, rates, schedule = _world()
    result = run_backtest(
        CoveredCall(target_delta=0.25), close, sigma, rates, schedule, ticker="S"
    )
    result.assert_identity()  # clean to start with

    result.bars.loc[result.bars.index[500], "residual"] = float("nan")
    with pytest.raises(AssertionError, match="NaN residual"):
        result.assert_identity()


@pytest.mark.parametrize("bad", [0.0, -50.0])
def test_non_positive_starting_equity_is_refused(bad):
    """The collateralisation guard reads `ctx.equity > 0` and keeps its stale share count when
    that is false, so it fails OPEN. At starting_equity=0 the book held a full share against
    cash of -101.02 and wrote a call against it, with the identity passing."""
    close, sigma, rates, schedule = _world()
    with pytest.raises(ValueError, match="starting_equity must be positive"):
        run_backtest(
            CoveredCall(target_delta=0.25), close, sigma, rates, schedule,
            starting_equity=bad, ticker="S",
        )


def test_bar_zero_is_no_longer_exempt_from_the_identity():
    """Bar 0's residual was hardcoded to zero, exempting initialisation from the only check in
    the engine. Every bar-0 trade is value-neutral, so value must equal the opening equity
    exactly, which is a real invariant rather than an assumption."""
    close, sigma, rates, schedule = _world()
    opening = float(close.iloc[0]) * 3.0
    for strategy in (BuyHold(), CoveredCall(target_delta=0.25), ProtectivePut(strike_rule="moneyness", otm_pct=0.05)):
        result = run_backtest(
            strategy, close, sigma, rates, schedule,
            starting_equity=opening, ticker="S",
        )
        assert result.bars["residual"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert result.value.iloc[0] == pytest.approx(opening, abs=1e-9)
        result.assert_identity()


def test_duplicate_index_is_refused():
    """position_of_index keeps the LAST occurrence, so a duplicated expiry would settle at
    time value instead of intrinsic and bank a mark above intrinsic."""
    close, sigma, rates, schedule = _world()
    dupe = close.index[100]
    close2 = pd.concat([close, close.loc[[dupe]]]).sort_index()
    sigma2 = pd.concat([sigma, sigma.loc[[dupe]]]).sort_index()
    rates2 = pd.concat([rates, rates.loc[[dupe]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate dates"):
        run_backtest(BuyHold(), close2, sigma2, rates2, schedule, ticker="S")
