"""The wheel: cash-secured puts until assigned, then covered calls until called away.

The only strategy here with genuine state. It alternates between two phases:

  CASH  hold no stock, write a cash-secured put. Expires worthless, keep the premium and
        write another. Finishes in the money, the shares are put to you.
  STOCK hold the shares, write a covered call. Expires worthless, keep the premium and write
        another. Finishes in the money, the shares go and you are back to cash.

Its appeal is collecting premium in both phases while only ever transacting stock at a price
already agreed. Its risk is not symmetric with that story: assignment happens precisely when
the trade has gone against you, so the wheel accumulates stock in falling markets and sheds
it in rising ones. On a sustained downtrend that is the worst possible way to be long. On a
range-bound name it is close to ideal.

Phase transitions are driven by what the ledger actually settled, via the engine's
`note_expiry` hook, not by a variable the strategy maintains in parallel. Mutable state that
can silently disagree with the ledger is how the handout lost an expiring position's payoff.
"""

from __future__ import annotations

from typing import Literal

from .base import BarContext, OpenPosition, OptionSpec, contracts_for, resolve_strike

StrikeRule = Literal["delta", "moneyness"]
#: Below this many shares the position is flat. Guards float dust from a rebalance, so a
#: residual 1e-13 shares does not read as the stock phase.
FLAT_TOLERANCE = 1e-9


class Wheel:
    def __init__(
        self,
        strike_rule: StrikeRule = "delta",
        target_delta: float = 0.25,
        put_otm: float = 0.05,
        call_otm: float = 0.05,
    ) -> None:
        if strike_rule not in ("delta", "moneyness"):
            raise ValueError(f"unknown strike_rule {strike_rule!r}")
        if not 0.0 < target_delta < 1.0:
            raise ValueError(f"target_delta must be in (0,1), got {target_delta!r}")
        for label, value in (("put_otm", put_otm), ("call_otm", call_otm)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{label} must be in [0,1), got {value!r}")
        self.strike_rule = strike_rule
        self.target_delta = float(target_delta)
        self.put_otm = float(put_otm)
        self.call_otm = float(call_otm)

        self._in_stock = False
        self._shares_held = 0.0
        #: Counted so the phase mix is reportable rather than inferred.
        self.put_writes = 0
        self.call_writes = 0
        self.assignments_taken = 0
        self.called_away = 0

    @property
    def name(self) -> str:
        if self.strike_rule == "delta":
            return f"wheel_delta_{self.target_delta:.2f}"
        return f"wheel_otm_{self.put_otm:.0%}_{self.call_otm:.0%}"

    def note_expiry(self, spot: float, settled: tuple[OpenPosition, ...]) -> None:
        """Read the phase transition off what actually settled."""
        for pos in settled:
            in_the_money = (
                spot < pos.strike if pos.kind == "put" else spot > pos.strike
            )
            if pos.kind == "put" and pos.quantity < 0 and in_the_money:
                self._in_stock = True  # the shares were put to us
                self.assignments_taken += 1
            elif pos.kind == "call" and pos.quantity < 0 and in_the_money:
                self._in_stock = False  # the shares were called away
                self.called_away += 1

    def target_shares(self, ctx: BarContext) -> float:
        if ctx.open_positions:
            return self._shares_held  # never resize while an option is written against it
        if ctx.spot <= 0 or ctx.equity <= 0:
            return self._shares_held
        self._shares_held = ctx.equity / ctx.spot if self._in_stock else 0.0
        return self._shares_held

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []

        if self._in_stock:
            strike = resolve_strike(
                ctx, "call", self.strike_rule, self.target_delta, self.call_otm
            )
            if strike is None or self._shares_held <= FLAT_TOLERANCE:
                return []
            self.call_writes += 1
            return [OptionSpec("call", strike, ctx.expiry, -self._shares_held)]

        strike = resolve_strike(
            ctx, "put", self.strike_rule, self.target_delta, self.put_otm
        )
        if strike is None:
            return []
        contracts = contracts_for(ctx.equity, strike)
        if contracts <= 0:
            return []
        self.put_writes += 1
        return [OptionSpec("put", strike, ctx.expiry, -contracts)]
