"""Covered call: hold one share, write one out-of-the-money call each expiry cycle.

Why it can beat buying and holding. The writer sells the right tail of the distribution
and is paid for it. Two things make that trade profitable on average rather than merely
different:

  1. Implied volatility trades above subsequent realised volatility most of the time.
     The premium is priced off the former and the loss is driven by the latter, so the
     seller collects the spread. That is the variance risk premium.
  2. Selling the tail removes the largest contributors to variance while removing only a
     capped amount of return. Sharpe is a ratio, so shrinking the denominator faster
     than the numerator raises it even when total return falls.

What it costs. Every rally beyond the strike is forfeited. On a strong uptrend the
overlay loses to buy-and-hold on total return and can only win on a risk-adjusted
basis. That is a real property of the strategy, not a modelling artefact, and it should
be reported rather than tuned away.

Two ways to pick the strike, both pre-specified rather than fitted:
  moneyness  strike a fixed percentage above spot. Simple, but its delta drifts with
             volatility, so the assignment probability is not stable.
  delta      strike at a fixed option delta. Holds the assignment probability roughly
             constant across volatility regimes, which is what a real overlay desk does.
"""

from __future__ import annotations

from typing import Literal

from ..options.bs import strike_from_delta
from .base import BarContext, OptionSpec

StrikeRule = Literal["delta", "moneyness"]


class CoveredCall:
    def __init__(
        self,
        strike_rule: StrikeRule = "delta",
        target_delta: float = 0.25,
        otm_pct: float = 0.05,
        shares: float = 1.0,
        contracts: float = 1.0,
    ) -> None:
        if strike_rule not in ("delta", "moneyness"):
            raise ValueError(f"unknown strike_rule {strike_rule!r}")
        if not 0.0 < target_delta < 1.0:
            raise ValueError(f"target_delta must be in (0,1), got {target_delta!r}")
        if otm_pct <= -1.0:
            raise ValueError(f"otm_pct must be > -1, got {otm_pct!r}")
        if contracts <= 0:
            raise ValueError("contracts must be positive; the call is written, not bought")
        if contracts > shares:
            raise ValueError(
                f"writing {contracts} call(s) against {shares} share(s) is not covered"
            )
        self.strike_rule = strike_rule
        self.target_delta = float(target_delta)
        self.otm_pct = float(otm_pct)
        self.shares = float(shares)
        self.contracts = float(contracts)

    @property
    def name(self) -> str:
        if self.strike_rule == "delta":
            return f"covered_call_delta_{self.target_delta:.2f}"
        return f"covered_call_otm_{self.otm_pct:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return self.shares

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.years_to_expiry <= 0.0:
            return []
        # Never stack a second call on top of a live one; that would be uncovered.
        if any(p.kind == "call" and p.quantity < 0 for p in ctx.open_positions):
            return []

        # ctx.sigma is the pricing (implied) vol the engine also marks with.
        sigma = ctx.sigma
        if not sigma > 0.0:
            return []

        if self.strike_rule == "moneyness":
            strike = round(ctx.spot * (1.0 + self.otm_pct), 2)
        else:
            strike = round(
                strike_from_delta(
                    ctx.spot,
                    ctx.years_to_expiry,
                    ctx.rate,
                    sigma,
                    self.target_delta,
                    "call",
                ),
                2,
            )

        return [
            OptionSpec(
                kind="call",
                strike=strike,
                expiry=ctx.expiry,
                quantity=-self.contracts,  # short
            )
        ]
