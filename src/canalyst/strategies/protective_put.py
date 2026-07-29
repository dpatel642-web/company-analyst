"""Protective put: hold one share, buy a downside put each cycle. Comparison arm.

The mirror image of the covered call. It buys the left tail rather than selling the
right one, so it pays the variance risk premium instead of collecting it. Drawdowns
shrink, which can lift Sharpe, but the premium is a persistent drag and total return
usually lands below buy-and-hold.

It is here to make the covered call's result legible. If both overlays beat the
benchmark on total return, something is wrong with the engine rather than right with
the strategies.
"""

from __future__ import annotations

from typing import Literal

from ..options.bs import bs_price, strike_from_delta
from .base import BarContext, OptionSpec

StrikeRule = Literal["delta", "moneyness"]


class ProtectivePut:
    def __init__(
        self,
        strike_rule: StrikeRule = "moneyness",
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
        if not 0.0 < otm_pct < 1.0:
            raise ValueError(f"otm_pct must be in (0,1) for a put, got {otm_pct!r}")
        if contracts <= 0:
            raise ValueError("contracts must be positive")
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
            return f"protective_put_delta_{self.target_delta:.2f}"
        return f"protective_put_otm_{self.otm_pct:.0%}"

    def _strike_for(self, ctx: BarContext) -> float | None:
        """The put strike this bar would use, or None if it cannot be determined."""
        if ctx.expiry is None or ctx.years_to_expiry <= 0.0 or not ctx.sigma > 0.0:
            return None
        if self.strike_rule == "moneyness":
            return round(ctx.spot * (1.0 - self.otm_pct), 2)
        try:
            return round(
                strike_from_delta(
                    ctx.spot, ctx.years_to_expiry, ctx.rate, ctx.sigma,
                    self.target_delta, "put", q=ctx.div_yield,
                ),
                2,
            )
        except ValueError:
            return None

    def target_shares(self, ctx: BarContext) -> float:
        if not self.fully_collateralised:
            return self.shares
        if not ctx.open_positions and ctx.spot > 0 and ctx.equity > 0:
            # Reserve the put's cost BEFORE sizing the share leg. Sweeping all of equity
            # into stock and then buying protection finances the premium at the risk-free
            # rate, which is a small levered tailwind for the one arm whose job is to be a
            # sanity check. Measured before this fix: cash negative on 85% of bars.
            # Solving `shares * (spot + premium_per_share) == equity` makes the stock and
            # its insurance fit inside the capital together.
            cost = 0.0
            if ctx.is_roll:
                strike = self._strike_for(ctx)
                if strike is not None:
                    cost = bs_price(
                        ctx.spot, strike, ctx.years_to_expiry, ctx.rate, ctx.sigma,
                        "put", q=ctx.div_yield,
                    )
            self._shares_held = ctx.equity / (ctx.spot + max(cost, 0.0))
        return self._shares_held

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.years_to_expiry <= 0.0:
            return []
        if any(p.kind == "put" and p.quantity > 0 for p in ctx.open_positions):
            return []

        strike = self._strike_for(ctx)
        if strike is None:
            return []

        return [
            OptionSpec(
                kind="put",
                strike=strike,
                expiry=ctx.expiry,
                quantity=+(
                    self._shares_held if self.fully_collateralised else self.contracts
                ),  # long
            )
        ]
