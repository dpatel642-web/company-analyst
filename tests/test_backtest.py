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
from canalyst.options.bs import bs_price
from canalyst.options.vol import apply_markup, close_to_close
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


def test_interest_accrues_on_cash_and_matters():
    """Extra starting equity leaves idle cash, which must earn the prevailing rate."""
    close, sigma, rates, schedule = _world(drift=0.0)
    idle = float(close.iloc[0]) * 3.0  # well above one share, so cash stays positive
    at_zero = run_backtest(
        BuyHold(), close, sigma, pd.Series(0.0, index=close.index), schedule,
        starting_equity=idle, ticker="S",
    )
    at_five = run_backtest(
        BuyHold(), close, sigma, pd.Series(0.05, index=close.index), schedule,
        starting_equity=idle, ticker="S",
    )
    assert at_five.bars["interest"].sum() > at_zero.bars["interest"].sum()
    assert at_zero.bars["interest"].sum() == pytest.approx(0.0, abs=1e-12)


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
    premium = bs_price(
        100.0,
        strike,
        (idx.get_loc(expiry_day) - idx.get_loc(roll_day)) / TRADING_DAYS,
        0.04,
        0.30,
        "call",
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
