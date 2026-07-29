"""Daily mark-to-market backtest engine, built around one invariant.

THE INVARIANT

Portfolio value can only change for reasons that are named. On every bar:

    value_t - value_{t-1} ==   shares_{t-1} * (spot_t - spot_{t-1})   share P&L
                             + shares_{t-1} * dividend_t              dividends
                             + interest on cash_{t-1}                 interest
                             + sum over positions held t-1 -> t of
                                   (mark_t - mark_{t-1})              option MTM
                             - fees_t

Every term is recorded per bar and the residual is stored. `assert_identity` fails if
any bar's residual exceeds tolerance.

Transactions are absent from that list on purpose. Opening an option at its fair value
moves value from cash into the position and back out again, netting zero; so does
settling one at intrinsic, and so does buying a share at spot. A transaction that is
*not* value-neutral means either mispricing or a leak, and the residual catches it.

WHY THIS SHAPE

The handout this replaces has two bugs that this invariant makes impossible:

  Bug A: it computed `value = shares * close + cash` where cash moved only on roll and
  expiry days. A short call's liability was invisible between them, so the option's
  entire P&L arrived as one spike on expiry day. Terminal value survived; daily
  volatility, and therefore Sharpe, did not. Here every open position is marked on
  every bar, and a missing mark shows up as a residual rather than as a plausible
  number.

  Bug B: on a day that was both an expiry and a roll, it wrote the expiring options'
  intrinsic value into a cell and then overwrote that cell with the *new* options'
  marks, so the expiring payoff evaporated every month. Here settlement credits cash
  before the new position is opened, and the two cannot alias because positions are
  objects in a list rather than columns in a frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .options.bs import bs_price
from .options.crosscheck import crr_price
from .options.vol import trailing_dividend_yield
from .strategies.base import BarContext, OpenPosition, OptionSpec, Strategy

TRADING_DAYS = 252
#: Residual tolerance per bar, in currency units. Floating point noise on values of
#: order 1e3 lands far below this; a real leak lands far above.
IDENTITY_TOL = 1e-7
#: Lattice steps for the independent mark check. Enough for sub-percent agreement with the
#: closed form without making the check too slow to leave switched on.
MARK_CHECK_STEPS = 600
#: Relative tolerance between the closed-form mark and the lattice. Generous on purpose: it
#: needs to pass ordinary discretisation error and fail a genuinely wrong input. A mark
#: carrying no theta is off by tens of percent, so 1% is not a close call.
MARK_CHECK_TOL = 0.01
#: Below this premium the relative comparison is meaningless, so the bar is skipped.
MARK_CHECK_MIN_PREMIUM = 1e-4
#: A mark error must ALSO be material against spot before it counts as a failure.
#:
#: Relative-to-premium alone produces false positives: a cheap out-of-the-money put carries a
#: small premium, so ordinary lattice discretisation error is a large FRACTION of it while
#: being economically irrelevant. Measured on real WMT data the protective put reached 0.938%
#: of premium, nearly tripping a 1% tolerance on nothing but discretisation. Two basis points
#: of spot is well below anything that moves an equity curve, and far below the tens of percent
#: a genuinely wrong mark produces.
MARK_CHECK_SPOT_TOL = 0.0002


@dataclass
class BacktestResult:
    strategy: str
    ticker: str
    bars: pd.DataFrame
    fees_paid: float
    rolls: int
    assignments: int
    #: Net cash received for options written, less cash paid for options bought.
    net_premium: float = 0.0
    #: Sampled interim marks repriced against an independent lattice. See `assert_marks`.
    mark_checks: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def value(self) -> pd.Series:
        return self.bars["value"]

    @property
    def max_abs_residual(self) -> float:
        return float(self.bars["residual"].abs().max())

    def assert_marks(self, tol: float = MARK_CHECK_TOL) -> None:
        """Raise if any sampled interim mark disagrees with an independent lattice price.

        WHY THIS EXISTS, AND WHY THE IDENTITY CANNOT REPLACE IT

        The accounting identity is a TELESCOPING sum. Over a position's life
        `sum(mark_t - mark_{t-1})` collapses to `mark_settle - mark_open`, so every interim
        mark cancels out. A wrong interim time-to-expiry therefore changes the daily return
        path, volatility, Sharpe and drawdown while leaving terminal value bit-identical and
        every residual at 1e-13.

        That is not hypothetical. Four mutants touching only the interim marking line were run
        against the 27 load-bearing assertions in the suite. An interim mark carrying ZERO
        THETA, no time decay at all, failed 0 of 27 while moving Sharpe by 0.024 and annualised
        volatility by 1.3 percentage points. The identity is a closure check on the ledger,
        evaluated with the model's own marks on both sides of every trade, so booking at price
        `p` and marking at the same `p` nets to zero for ANY `p`.

        This check pins the mark LEVEL instead, by repricing sampled positions with a
        Cox-Ross-Rubinstein lattice. The lattice shares no arithmetic with the closed form, so
        it also catches a wrong pricing argument, including a missing dividend yield.
        """
        if not self.mark_checks:
            return
        # Both conditions: a large fraction of a negligible premium is not a real problem.
        bad = [
            c for c in self.mark_checks
            if c["relative_error"] > tol and c["spot_error"] > MARK_CHECK_SPOT_TOL
        ]
        if bad:
            worst = max(bad, key=lambda c: c["spot_error"])
            raise AssertionError(
                f"interim mark disagrees with an independent lattice on "
                f"{len(bad)} of {len(self.mark_checks)} sampled bars. Worst on "
                f"{worst['date']:%Y-%m-%d}: closed form {worst['analytic']:.6f} vs lattice "
                f"{worst['lattice']:.6f}, relative error {worst['relative_error']:.4%} "
                f"(tolerance {tol:.2%}). The accounting identity cannot see this, because "
                f"interim marks telescope out of it."
            )

    @property
    def worst_mark_error(self) -> float:
        return max((c["relative_error"] for c in self.mark_checks), default=0.0)

    def assert_identity(self, tol: float = IDENTITY_TOL) -> None:
        """Raise unless every bar's value change is fully explained."""
        residual = self.bars["residual"]
        # pandas .max() is skipna=True, so an all-NaN residual yields nan, and `nan > tol` is
        # False. The documented last line of defence silently certified any run whose ledger
        # had gone NaN: one NaN in the rate series propagated through interest into cash, and
        # `summarise`'s dropna() then amputated 17 months while both checks passed.
        missing = int(residual.isna().sum())
        if missing:
            first = residual.index[residual.isna()][0]
            raise AssertionError(
                f"{missing} bar(s) have a NaN residual, first {first:%Y-%m-%d}. A NaN cannot "
                "be compared against a tolerance, so the identity cannot vouch for this run."
            )
        worst = residual.abs()
        if worst.max() > tol:
            day = worst.idxmax()
            row = self.bars.loc[day]
            raise AssertionError(
                f"accounting identity violated on {day:%Y-%m-%d}: "
                f"residual {row['residual']:.3e} exceeds {tol:.1e}\n"
                f"  d_value={row['d_value']:.6f} share_pnl={row['share_pnl']:.6f} "
                f"dividends={row['dividends']:.6f} interest={row['interest']:.6f} "
                f"option_mtm_change={row['option_mtm_change']:.6f} "
                f"fees={row['fees']:.6f}"
            )


def _mark(
    position_kind: str,
    strike: float,
    spot: float,
    years: float,
    rate: float,
    sigma: float,
    quantity: float,
    div_yield: float = 0.0,
) -> float:
    """Signed market value of a position. Short positions mark negative.

    `div_yield` is not optional in spirit. Pricing with q=0 on a dividend payer while the
    share leg separately banks the dividend pays the option writer twice.
    """
    unit = bs_price(
        spot, strike, max(years, 0.0), rate, max(sigma, 0.0), position_kind,
        q=max(div_yield, 0.0),
    )
    return quantity * unit


def run_backtest(
    strategy: Strategy,
    close: pd.Series,
    sigma: pd.Series,
    rate: pd.Series,
    schedule: pd.DataFrame,
    dividends: pd.Series | None = None,
    div_yield: pd.Series | None = None,
    starting_equity: float | None = None,
    fee_per_contract: float = 0.0,
    verify_marks: int = 0,
    ticker: str = "",
    settings: dict | None = None,
) -> BacktestResult:
    """Run `strategy` over the given bars.

    `close` must be the split-adjusted, dividend-unadjusted series: strikes are struck
    on it and assignment is decided against it. `schedule` is indexed by roll date with
    an `expiry` column.

    `sigma` is the **pricing** volatility, meaning the market's implied vol, and it
    governs the entire option lifecycle: the price paid or received on opening, and every
    subsequent daily mark. It is deliberately a separate input from whatever the
    underlying actually realises, because the gap between the two *is* the variance risk
    premium and therefore the whole reason writing options pays. Pass a realised-vol
    estimate and you have set that premium to zero, which makes writing a call close to
    a fair bet; pass `apply_markup(realised, 1.15)` and you are modelling a seller who
    collects the spread. Neither is wrong, but they answer different questions and the
    difference must be reported, never assumed.

    Per-share accounting throughout: one share of exposure, so the equity curve reads
    directly against buy-and-hold with no scaling factor.

    `starting_equity` is the capital the portfolio begins with, defaulting to one share's
    price so the curve starts level with buy-and-hold and returns are directly comparable.

    `verify_marks` samples that many interim bars and reprices every open position with an
    independent lattice, checked via `BacktestResult.assert_marks`. Off by default because it
    is roughly ten times the cost of a plain run, and a check that taxes every call is a check
    people switch off. Every real analysis path turns it on; see `assert_marks` for why the
    accounting identity cannot substitute for it.
    """
    index = pd.DatetimeIndex(close.index)
    if len(index) == 0:
        raise ValueError("cannot backtest an empty price series")
    if not index.equals(pd.DatetimeIndex(sigma.index)):
        raise ValueError("sigma index does not match close index")
    if not index.equals(pd.DatetimeIndex(rate.index)):
        raise ValueError("rate index does not match close index")
    # Reject NaN at the door. Every downstream check is comparison-based, and comparisons
    # against NaN are False, so a NaN input does not fail loudly, it disables the guards.
    for label, series in (("close", close), ("sigma", sigma), ("rate", rate)):
        bad = int(pd.Series(series).isna().sum())
        if bad:
            raise ValueError(
                f"{label} contains {bad} NaN value(s); refusing to run. A NaN propagates "
                "silently through the ledger and turns every tolerance check into a pass."
            )
    if index.has_duplicates:
        # position_of_index keeps the last occurrence, so a duplicated expiry would settle at
        # time value instead of intrinsic.
        raise ValueError("close index contains duplicate dates")

    dividends = (
        pd.Series(0.0, index=index) if dividends is None else dividends.reindex(index).fillna(0.0)
    )
    if div_yield is None:
        # Derive it rather than defaulting to zero: a silent q=0 on a dividend payer is a
        # one-signed transfer to the option writer, not a neutral simplification.
        div_yield = trailing_dividend_yield(dividends, close)
    div_yield = div_yield.reindex(index).ffill().fillna(0.0)
    expiry_by_roll = schedule["expiry"].to_dict() if len(schedule) else {}
    position_of_index = {d: i for i, d in enumerate(index)}
    dt = 1.0 / TRADING_DAYS

    # Bars on which to independently reprice open positions. Spread across the run rather
    # than clustered, and interim only: the open and expiry bars are already pinned by the
    # identity, so checking them would prove nothing about the marks in between.
    stride = max(1, len(index) // max(verify_marks, 1)) if verify_marks > 0 else 0
    mark_checks: list[dict] = []

    opening_equity = (
        float(starting_equity) if starting_equity is not None else float(close.iloc[0])
    )
    if not opening_equity > 0:
        # The collateralisation guard reads `ctx.equity > 0` and keeps its stale share count
        # when that is false, so it fails OPEN: at starting_equity=0 the book held a full
        # share against cash of -101.02 and wrote a call against it, identity passing.
        raise ValueError(
            f"starting_equity must be positive, got {opening_equity!r}. A non-positive "
            "opening balance makes the collateralisation guard fail open into leverage."
        )

    shares = 0.0
    cash = opening_equity
    book: list[OpenPosition] = []

    prev_value: float | None = None
    prev_spot: float | None = None
    rows: list[dict] = []
    fees_paid = 0.0
    net_premium = 0.0
    rolls = 0
    assignments = 0

    for i, day in enumerate(index):
        spot = float(close.iloc[i])
        vol = float(sigma.iloc[i])
        r = float(rate.iloc[i])
        div_per_share = float(dividends.iloc[i])
        q = float(div_yield.iloc[i])

        shares_at_open = shares
        cash_at_open = cash

        # ---- 1. carry: share P&L, dividends, interest on yesterday's cash
        share_pnl = 0.0 if prev_spot is None else shares_at_open * (spot - prev_spot)
        dividend_cash = shares_at_open * div_per_share
        interest = 0.0 if prev_value is None else cash_at_open * (math.exp(r * dt) - 1.0)
        cash += dividend_cash + interest

        # ---- 2. mark every open position at today's market, before any trading
        marked: list[OpenPosition] = []
        option_mtm_change = 0.0
        for pos in book:
            years = max(
                (position_of_index[pos.expiry] - i) / TRADING_DAYS, 0.0
            ) if pos.expiry in position_of_index else 0.0
            mark = _mark(
                pos.kind, pos.strike, spot, years, r, vol, pos.quantity, div_yield=q
            )
            if (
                stride
                and i % stride == 0
                and years > 0.0
                and pos.opened_on < day
                and abs(mark) > MARK_CHECK_MIN_PREMIUM
            ):
                mark_checks.append(
                    {
                        "date": day, "kind": pos.kind, "strike": pos.strike, "spot": spot,
                        "years": years, "rate": r, "sigma": vol, "div_yield": q,
                        "quantity": pos.quantity, "analytic": mark,
                    }
                )
            option_mtm_change += mark - pos.last_mark
            marked.append(
                OpenPosition(spec=pos.spec, opened_on=pos.opened_on, last_mark=mark)
            )
        book = marked

        # ---- 3. settle anything expiring today. Its mark is already intrinsic
        #         (years == 0), so crediting cash and dropping it is value-neutral.
        survivors: list[OpenPosition] = []
        settled: list[OpenPosition] = []
        for pos in book:
            if pos.expiry <= day:
                cash += pos.last_mark
                settled.append(pos)
                if abs(pos.last_mark) > 1e-12:
                    assignments += 1
            else:
                survivors.append(pos)
        book = survivors

        # Optional hook. A strategy with phases (the wheel) cannot otherwise tell "expired
        # worthless" from "was assigned", because both leave an empty book. Passing the
        # settled positions keeps the ledger the single source of truth rather than asking
        # the strategy to track a parallel variable that can disagree with it.
        if settled and hasattr(strategy, "note_expiry"):
            strategy.note_expiry(spot, tuple(settled))

        # ---- 4. let the strategy act
        is_roll = day in expiry_by_roll
        expiry = expiry_by_roll.get(day)
        years_to_expiry = (
            (position_of_index[expiry] - i) / TRADING_DAYS
            if expiry is not None and expiry in position_of_index
            else 0.0
        )
        # Equity available to trade with right now: the share leg, cash, and whatever
        # positions survived settlement. Strategies size against this so they cannot
        # lever up on their own losses.
        pre_trade_equity = shares * spot + cash + sum(p.last_mark for p in book)

        ctx = BarContext(
            date=day,
            spot=spot,
            sigma=vol,
            rate=r,
            dividend=div_per_share,
            div_yield=q,
            is_roll=is_roll,
            expiry=expiry,
            years_to_expiry=years_to_expiry,
            open_positions=tuple(book),
            equity=pre_trade_equity,
        )

        # share leg, traded at spot and therefore value-neutral
        target = float(strategy.target_shares(ctx))
        if not math.isclose(target, shares, rel_tol=0.0, abs_tol=1e-12):
            cash -= (target - shares) * spot
            shares = target

        # option leg, opened at fair value and therefore also value-neutral
        fees_today = 0.0
        for spec in strategy.options_to_open(ctx):
            years = (
                (position_of_index[spec.expiry] - i) / TRADING_DAYS
                if spec.expiry in position_of_index
                else 0.0
            )
            mark = _mark(
                spec.kind, spec.strike, spot, years, r, vol, spec.quantity, div_yield=q
            )
            cash -= mark  # short position: mark is negative, so cash rises
            net_premium += -mark
            book.append(OpenPosition(spec=spec, opened_on=day, last_mark=mark))
            # Per CONTRACT, not per spec. Charging once per leg understated costs by the
            # position size: at 100 contracts over 35 rolls it billed 22.75 instead of
            # 2275.00, a 100x error invisible at unit size and appearing the moment anyone
            # scales the notional.
            fees_today += fee_per_contract * abs(spec.quantity)
        if fees_today:
            cash -= fees_today
            fees_paid += fees_today
        if is_roll:
            rolls += 1

        # ---- 5. close the books and check that nothing went missing
        option_mtm = sum(p.last_mark for p in book)
        value = shares * spot + cash + option_mtm

        if prev_value is None:
            # Bar 0 used to hardcode a zero residual, which exempted initialisation from the
            # only check in the engine and is what let a non-positive opening balance run at
            # infinite leverage unnoticed. Every bar-0 trade is value-neutral (shares bought
            # at spot, options opened at fair value), so value must still equal the opening
            # equity exactly. That is a real invariant, so check it.
            d_value = np.nan
            residual = value - opening_equity
        else:
            d_value = value - prev_value
            explained = (
                share_pnl + dividend_cash + interest + option_mtm_change - fees_today
            )
            residual = d_value - explained

        rows.append(
            {
                "date": day,
                "spot": spot,
                "sigma": vol,
                "rate": r,
                "shares": shares,
                "cash": cash,
                "option_mtm": option_mtm,
                "value": value,
                "d_value": d_value,
                "share_pnl": share_pnl,
                "dividends": dividend_cash,
                "interest": interest,
                "option_mtm_change": option_mtm_change,
                "fees": fees_today,
                "residual": residual,
                "open_positions": len(book),
                "is_roll": is_roll,
            }
        )
        prev_value = value
        prev_spot = spot

    # Reprice the sampled marks with a lattice. Done after the loop so the cost never sits
    # inside the hot path, and so a slow check cannot distort anything it is measuring.
    for check in mark_checks:
        lattice = crr_price(
            check["spot"], check["strike"], check["years"], check["rate"],
            max(check["sigma"], 1e-9), check["kind"], q=check["div_yield"],
            steps=MARK_CHECK_STEPS,
        ) * check["quantity"]
        check["lattice"] = lattice
        gap = abs(check["analytic"] - lattice)
        scale = max(abs(check["analytic"]), abs(lattice), MARK_CHECK_MIN_PREMIUM)
        check["relative_error"] = gap / scale
        # Also express it against spot, so a large fraction of a tiny premium is not mistaken
        # for an error that matters to the portfolio.
        check["spot_error"] = gap / max(check["spot"], MARK_CHECK_MIN_PREMIUM)

    bars = pd.DataFrame(rows).set_index("date")
    return BacktestResult(
        strategy=strategy.name,
        ticker=ticker,
        bars=bars,
        fees_paid=fees_paid,
        net_premium=net_premium,
        mark_checks=mark_checks,
        rolls=rolls,
        assignments=assignments,
        settings=settings or {},
    )
