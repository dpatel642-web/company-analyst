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
from .strategies.base import BarContext, OpenPosition, OptionSpec, Strategy

TRADING_DAYS = 252
#: Residual tolerance per bar, in currency units. Floating point noise on values of
#: order 1e3 lands far below this; a real leak lands far above.
IDENTITY_TOL = 1e-7


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
    settings: dict = field(default_factory=dict)

    @property
    def value(self) -> pd.Series:
        return self.bars["value"]

    @property
    def max_abs_residual(self) -> float:
        return float(self.bars["residual"].abs().max())

    def assert_identity(self, tol: float = IDENTITY_TOL) -> None:
        """Raise unless every bar's value change is fully explained."""
        worst = self.bars["residual"].abs()
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
) -> float:
    """Signed market value of a position. Short positions mark negative."""
    unit = bs_price(spot, strike, max(years, 0.0), rate, max(sigma, 0.0), position_kind)
    return quantity * unit


def run_backtest(
    strategy: Strategy,
    close: pd.Series,
    sigma: pd.Series,
    rate: pd.Series,
    schedule: pd.DataFrame,
    dividends: pd.Series | None = None,
    starting_cash: float = 0.0,
    fee_per_contract: float = 0.0,
    auto_fund: bool = True,
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

    `auto_fund` credits exactly enough cash on the first bar to pay for the opening
    share position. Without it the purchase is debited from nothing, the curve starts at
    zero instead of the share price, and every return is undefined on the first bar.
    Funding it also gets the interest right: cash spent on the share stops earning, while
    premium subsequently collected does earn.
    """
    index = pd.DatetimeIndex(close.index)
    if len(index) == 0:
        raise ValueError("cannot backtest an empty price series")
    if not index.equals(pd.DatetimeIndex(sigma.index)):
        raise ValueError("sigma index does not match close index")
    if not index.equals(pd.DatetimeIndex(rate.index)):
        raise ValueError("rate index does not match close index")

    dividends = (
        pd.Series(0.0, index=index) if dividends is None else dividends.reindex(index).fillna(0.0)
    )
    expiry_by_roll = schedule["expiry"].to_dict() if len(schedule) else {}
    position_of_index = {d: i for i, d in enumerate(index)}
    dt = 1.0 / TRADING_DAYS

    shares = 0.0
    cash = float(starting_cash)
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
            mark = _mark(pos.kind, pos.strike, spot, years, r, vol, pos.quantity)
            option_mtm_change += mark - pos.last_mark
            marked.append(
                OpenPosition(spec=pos.spec, opened_on=pos.opened_on, last_mark=mark)
            )
        book = marked

        # ---- 3. settle anything expiring today. Its mark is already intrinsic
        #         (years == 0), so crediting cash and dropping it is value-neutral.
        survivors: list[OpenPosition] = []
        for pos in book:
            if pos.expiry <= day:
                cash += pos.last_mark
                if abs(pos.last_mark) > 1e-12:
                    assignments += 1
            else:
                survivors.append(pos)
        book = survivors

        # ---- 4. let the strategy act
        is_roll = day in expiry_by_roll
        expiry = expiry_by_roll.get(day)
        years_to_expiry = (
            (position_of_index[expiry] - i) / TRADING_DAYS
            if expiry is not None and expiry in position_of_index
            else 0.0
        )
        ctx = BarContext(
            date=day,
            spot=spot,
            sigma=vol,
            rate=r,
            dividend=div_per_share,
            is_roll=is_roll,
            expiry=expiry,
            years_to_expiry=years_to_expiry,
            open_positions=tuple(book),
        )

        # share leg, traded at spot and therefore value-neutral
        target = float(strategy.target_shares(ctx))
        if i == 0 and auto_fund:
            # Credit the opening position's cost so the purchase nets to zero and the
            # curve starts at the share price rather than at zero.
            cash += target * spot
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
            mark = _mark(spec.kind, spec.strike, spot, years, r, vol, spec.quantity)
            cash -= mark  # short position: mark is negative, so cash rises
            net_premium += -mark
            book.append(OpenPosition(spec=spec, opened_on=day, last_mark=mark))
            fees_today += fee_per_contract
        if fees_today:
            cash -= fees_today
            fees_paid += fees_today
        if is_roll:
            rolls += 1

        # ---- 5. close the books and check that nothing went missing
        option_mtm = sum(p.last_mark for p in book)
        value = shares * spot + cash + option_mtm

        if prev_value is None:
            d_value = np.nan
            residual = 0.0
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

    bars = pd.DataFrame(rows).set_index("date")
    return BacktestResult(
        strategy=strategy.name,
        ticker=ticker,
        bars=bars,
        fees_paid=fees_paid,
        net_premium=net_premium,
        rolls=rolls,
        assignments=assignments,
        settings=settings or {},
    )
