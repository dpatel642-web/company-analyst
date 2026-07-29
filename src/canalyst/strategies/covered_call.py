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

FULL COLLATERALISATION, and why it is not optional

The obvious way to model this is "hold one share forever, and on assignment pay out
(S_T - K) and keep the share". The handout does that, and it is wrong in a way that
inverts the result.

Each assignment debits cash without reducing the share position, so a run of losses
leaves a full share of exposure financed by a growing margin loan. Measured on TSLA over
five years, cash reached -240 against a share worth 222: exposure of 4.4x the remaining
equity. Volatility then comes out *higher* than the underlying's, which is impossible for
a genuine covered call, and the strategy is quietly being judged as a levered long.

A real buy-write is collateralised: after a loss you own fewer shares, not the same share
on credit. CBOE's BXM index works this way. So share count is reset to equity/spot at
each roll, which caps exposure at 1.0x and can only ever deleverage. For buy-and-hold
the same rule is a no-op, since equity/spot is identically one share.
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
        fully_collateralised: bool = True,
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
        self.fully_collateralised = bool(fully_collateralised)
        self._shares_held = float(shares)

    @property
    def name(self) -> str:
        if self.strike_rule == "delta":
            return f"covered_call_delta_{self.target_delta:.2f}"
        return f"covered_call_otm_{self.otm_pct:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        if not self.fully_collateralised:
            return self.shares
        # Rebalance only when no option is open against the shares. While one is, the
        # share count must stay fixed or the position stops being covered mid-cycle.
        # Keying on "book is empty" rather than "is a roll date" also catches the final
        # expiry, which has no successor roll and would otherwise be left unrebalanced
        # carrying its settlement debit as a margin loan.
        if not ctx.open_positions and ctx.spot > 0 and ctx.equity > 0:
            self._shares_held = ctx.equity / ctx.spot
        return self._shares_held

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
                    q=ctx.div_yield,
                ),
                2,
            )

        # Write against exactly the shares held, never more: that is what "covered" means.
        contracts = self._shares_held if self.fully_collateralised else self.contracts
        if contracts <= 0:
            return []

        return [
            OptionSpec(
                kind="call",
                strike=strike,
                expiry=ctx.expiry,
                quantity=-contracts,  # short
            )
        ]
