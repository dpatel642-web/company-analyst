"""Vertical spreads: defined-risk directional bets, holding no stock.

A bull call spread buys a call and sells a further one, so both the cost and the maximum
gain are capped. Maximum loss is the net debit, maximum gain is the strike width less that
debit.

Honest note on comparability. These are NOT overlays on a share position, so setting them
against buy and hold is closer to apples-to-oranges than the covered call is. A spread
commits only its debit and leaves the rest of the capital in cash earning the risk-free
rate, so its equity curve is mostly cash and its volatility is correspondingly small. That
makes Sharpe flattering in a way that says more about position sizing than about the
structure. Sizing here is to notional (equity/spot contracts) so the exposure is comparable
to holding the stock, and the residual cash genuinely earns the bill rate, which is the
fairest available framing rather than a perfect one.
"""

from __future__ import annotations

from typing import Literal

from .base import BarContext, OptionSpec, contracts_for, resolve_strike

Direction = Literal["bull_call", "bear_put"]


class VerticalSpread:
    def __init__(
        self,
        direction: Direction = "bull_call",
        long_otm: float = 0.00,
        short_otm: float = 0.10,
    ) -> None:
        if direction not in ("bull_call", "bear_put"):
            raise ValueError(f"unknown direction {direction!r}")
        if not -0.5 < long_otm < 1.0 or not -0.5 < short_otm < 1.0:
            raise ValueError("strike distances must be in (-0.5, 1.0)")
        if short_otm <= long_otm:
            raise ValueError(
                f"the short leg must be further out than the long leg, "
                f"got long={long_otm!r} short={short_otm!r}"
            )
        self.direction = direction
        self.long_otm = float(long_otm)
        self.short_otm = float(short_otm)

    @property
    def name(self) -> str:
        return f"{self.direction}_spread_{self.long_otm:.0%}_{self.short_otm:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return 0.0

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        kind = "call" if self.direction == "bull_call" else "put"
        long_k = resolve_strike(ctx, kind, "moneyness", 0.0, self.long_otm)
        short_k = resolve_strike(ctx, kind, "moneyness", 0.0, self.short_otm)
        if long_k is None or short_k is None or long_k == short_k:
            return []
        contracts = contracts_for(ctx.equity, ctx.spot)
        if contracts <= 0:
            return []
        return [
            OptionSpec(kind, long_k, ctx.expiry, +contracts),
            OptionSpec(kind, short_k, ctx.expiry, -contracts),
        ]
