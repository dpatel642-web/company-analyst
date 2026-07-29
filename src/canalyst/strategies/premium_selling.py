"""Short strangle and iron condor: selling both tails, holding no stock.

These are the purest expression of the variance risk premium. A short strangle writes a call
above and a put below and keeps both premiums if the underlying finishes between them, which
is why it is the structure most directly paid for the premium the covered call only partly
harvests.

⚠️ THE SHORT STRANGLE'S CALL LEG IS NAKED, AND THAT RISK IS UNBOUNDED.

There is no stock to deliver, so a large upward move has no cap on the loss. This is not a
modelling artefact to be smoothed over, it is the actual risk of the actual structure, and it
is the reason the position is sized to `equity / spot` here rather than to anything more
aggressive. Even so, a single sufficiently violent gap can exceed the collateral, and the
backtest will show that as a loss rather than as a margin call, which is more forgiving than
reality would be. Read short-strangle results with that in mind.

The iron condor buys further wings to cap both tails. It collects less premium and, unlike the
strangle, cannot lose more than the width between its strikes. That makes it the version a
risk desk would actually run, and the honest comparison between the two is what the wings
cost in return for making the loss finite.
"""

from __future__ import annotations

from .base import BarContext, OptionSpec, contracts_for, resolve_strike


class ShortStrangle:
    """Sell an OTM call and an OTM put. The call leg is naked; see the module docstring."""

    def __init__(self, call_otm: float = 0.10, put_otm: float = 0.10) -> None:
        for label, value in (("call_otm", call_otm), ("put_otm", put_otm)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{label} must be in (0,1), got {value!r}")
        self.call_otm = float(call_otm)
        self.put_otm = float(put_otm)

    @property
    def name(self) -> str:
        return f"short_strangle_{self.call_otm:.0%}_{self.put_otm:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return 0.0

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        call_k = resolve_strike(ctx, "call", "moneyness", 0.0, self.call_otm)
        put_k = resolve_strike(ctx, "put", "moneyness", 0.0, self.put_otm)
        if call_k is None or put_k is None:
            return []
        contracts = contracts_for(ctx.equity, ctx.spot)
        if contracts <= 0:
            return []
        return [
            OptionSpec("call", call_k, ctx.expiry, -contracts),
            OptionSpec("put", put_k, ctx.expiry, -contracts),
        ]


class IronCondor:
    """Short strangle with protective wings, so the maximum loss is finite."""

    def __init__(
        self,
        call_otm: float = 0.10,
        put_otm: float = 0.10,
        wing_width: float = 0.05,
    ) -> None:
        for label, value in (("call_otm", call_otm), ("put_otm", put_otm)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{label} must be in (0,1), got {value!r}")
        if not 0.0 < wing_width < 1.0:
            raise ValueError(f"wing_width must be in (0,1), got {wing_width!r}")
        self.call_otm = float(call_otm)
        self.put_otm = float(put_otm)
        self.wing_width = float(wing_width)

    @property
    def name(self) -> str:
        return f"iron_condor_{self.call_otm:.0%}_{self.put_otm:.0%}_w{self.wing_width:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        return 0.0

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        short_call = resolve_strike(ctx, "call", "moneyness", 0.0, self.call_otm)
        long_call = resolve_strike(
            ctx, "call", "moneyness", 0.0, self.call_otm + self.wing_width
        )
        short_put = resolve_strike(ctx, "put", "moneyness", 0.0, self.put_otm)
        long_put = resolve_strike(
            ctx, "put", "moneyness", 0.0, min(self.put_otm + self.wing_width, 0.95)
        )
        strikes = [short_call, long_call, short_put, long_put]
        if any(k is None for k in strikes) or long_call <= short_call:
            return []
        if long_put >= short_put:
            return []
        contracts = contracts_for(ctx.equity, ctx.spot)
        if contracts <= 0:
            return []
        return [
            OptionSpec("call", short_call, ctx.expiry, -contracts),
            OptionSpec("call", long_call, ctx.expiry, +contracts),
            OptionSpec("put", short_put, ctx.expiry, -contracts),
            OptionSpec("put", long_put, ctx.expiry, +contracts),
        ]
