"""Long straddle: buy a call and a put at the same strike. A bet on movement.

It profits when the underlying moves further than the combined premium implies, in either
direction, and loses the whole premium when it sits still.

It should be expected to LOSE under this project's pricing basis, and understanding why is
more useful than the number. Options here are priced off realised volatility, so the
straddle is bought at exactly the volatility the underlying goes on to realise. That makes
it a fair bet before costs and a losing one after any friction. A straddle only makes money
when implied volatility is *below* subsequent realised volatility, which is the opposite of
the variance risk premium the covered call harvests.

Reported anyway, because it is on the assignment's list and because a strategy set where
everything wins is a strategy set with a bug in it. The straddle is the control.
"""

from __future__ import annotations

from .base import BarContext, OptionSpec, contracts_for, resolve_strike


class LongStraddle:
    def __init__(self, moneyness: float = 0.00) -> None:
        if not -0.5 < moneyness < 1.0:
            raise ValueError(f"moneyness must be in (-0.5, 1.0), got {moneyness!r}")
        self.moneyness = float(moneyness)

    @property
    def name(self) -> str:
        return "long_straddle" if self.moneyness == 0.0 else f"long_strangle_{self.moneyness:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return 0.0

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        call_k = resolve_strike(ctx, "call", "moneyness", 0.0, self.moneyness)
        put_k = resolve_strike(ctx, "put", "moneyness", 0.0, self.moneyness)
        if call_k is None or put_k is None:
            return []
        contracts = contracts_for(ctx.equity, ctx.spot)
        if contracts <= 0:
            return []
        return [
            OptionSpec("call", call_k, ctx.expiry, +contracts),
            OptionSpec("put", put_k, ctx.expiry, +contracts),
        ]
