"""Cash-secured put: hold no stock, write a put, keep enough cash to be assigned.

Put-call parity says a cash-secured put and a covered call **at the same strike** have the
same payoff: both are `min(S_T, K)`. That equivalence is worth stating precisely, because it
is easy to over-apply and I did so in an earlier draft of this docstring.

A 0.25-delta cash-secured put and a 0.25-delta covered call are NOT parity equivalents. The
put's strike sits BELOW spot and the call's sits ABOVE it, so they are different strikes and
different trades. Their deltas are not comparable either: a short 0.25-delta put carries about
+0.25 delta, while a covered call carries roughly 1 - 0.25 = 0.75. The covered call is a
three-times-larger directional bet. Comparing them tests position sizing, not parity.

Where the structures do coincide in strike, two real differences remain:
  - The collateral sits in cash and earns the risk-free rate throughout. Over a window where
    the 13-week bill averaged 3.6%, that is a contributor, not a rounding error.
  - There is no stock position, so no dividend is received. On a payer that cuts the other way.

Sizing is to the strike, not to spot: the whole point of "cash-secured" is holding enough
cash to buy the shares if assigned. Sizing to spot would leave the position slightly
under-collateralised whenever the strike sits above the current price.
"""

from __future__ import annotations

from typing import Literal

from .base import BarContext, OptionSpec, contracts_for, resolve_strike

StrikeRule = Literal["delta", "moneyness"]


class CashSecuredPut:
    def __init__(
        self,
        strike_rule: StrikeRule = "delta",
        target_delta: float = 0.25,
        otm_pct: float = 0.05,
    ) -> None:
        if strike_rule not in ("delta", "moneyness"):
            raise ValueError(f"unknown strike_rule {strike_rule!r}")
        if not 0.0 < target_delta < 1.0:
            raise ValueError(f"target_delta must be in (0,1), got {target_delta!r}")
        if not 0.0 <= otm_pct < 1.0:
            raise ValueError(f"otm_pct must be in [0,1) for a put, got {otm_pct!r}")
        self.strike_rule = strike_rule
        self.target_delta = float(target_delta)
        self.otm_pct = float(otm_pct)

    @property
    def name(self) -> str:
        if self.strike_rule == "delta":
            return f"cash_secured_put_delta_{self.target_delta:.2f}"
        return f"cash_secured_put_otm_{self.otm_pct:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return 0.0  # cash and a short put, never stock

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        strike = resolve_strike(
            ctx, "put", self.strike_rule, self.target_delta, self.otm_pct
        )
        if strike is None:
            return []
        contracts = contracts_for(ctx.equity, strike)
        if contracts <= 0:
            return []
        return [
            OptionSpec(kind="put", strike=strike, expiry=ctx.expiry, quantity=-contracts)
        ]
