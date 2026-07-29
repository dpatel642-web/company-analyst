"""Collar: hold the stock, write a call above, buy a put below.

The combination of the two single-leg overlays already built, and the most defensive of the
stock-holding strategies. The written call pays for the protective put, so the running cost
is small or nil, and in exchange both tails are gone: no participation above the call
strike, no loss below the put strike.

`ZeroCostCollar` solves for the put strike at which the two premiums cancel, which is the
form most often quoted in practice because it needs no cash outlay. It is solved rather than
assumed: bisection on the put's distance out of the money until the net premium is within a
tolerance of zero. A fixed 5%/5% collar is NOT zero cost in general, because the volatility
skew prices downside protection above equidistant upside.

If no zero-cost put strike exists inside the search bracket, the strategy writes the call
alone rather than silently substituting a different structure. A collar that is not a collar
should be visible in the results, not hidden.
"""

from __future__ import annotations

from ..options.bs import bs_price
from .base import BarContext, OptionSpec, resolve_strike


class Collar:
    """Fixed-moneyness collar: call `call_otm` above spot, put `put_otm` below."""

    def __init__(self, call_otm: float = 0.05, put_otm: float = 0.05) -> None:
        if not 0.0 < call_otm < 1.0:
            raise ValueError(f"call_otm must be in (0,1), got {call_otm!r}")
        if not 0.0 < put_otm < 1.0:
            raise ValueError(f"put_otm must be in (0,1), got {put_otm!r}")
        self.call_otm = float(call_otm)
        self.put_otm = float(put_otm)
        self._shares_held = 1.0

    @property
    def name(self) -> str:
        return f"collar_{self.call_otm:.0%}_{self.put_otm:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        if not ctx.open_positions and ctx.spot > 0 and ctx.equity > 0:
            self._shares_held = ctx.equity / ctx.spot
        return self._shares_held

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        call_k = resolve_strike(ctx, "call", "moneyness", 0.0, self.call_otm)
        put_k = resolve_strike(ctx, "put", "moneyness", 0.0, self.put_otm)
        if call_k is None or put_k is None or self._shares_held <= 0:
            return []
        return [
            OptionSpec("call", call_k, ctx.expiry, -self._shares_held),
            OptionSpec("put", put_k, ctx.expiry, +self._shares_held),
        ]


class ZeroCostCollar:
    """Collar whose put strike is solved so the net premium is approximately zero."""

    def __init__(
        self,
        call_otm: float = 0.05,
        max_put_otm: float = 0.40,
        tolerance: float = 1e-4,
        max_iter: int = 60,
    ) -> None:
        if not 0.0 < call_otm < 1.0:
            raise ValueError(f"call_otm must be in (0,1), got {call_otm!r}")
        if not 0.0 < max_put_otm < 1.0:
            raise ValueError(f"max_put_otm must be in (0,1), got {max_put_otm!r}")
        self.call_otm = float(call_otm)
        self.max_put_otm = float(max_put_otm)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self._shares_held = 1.0
        #: Rolls where no zero-cost put existed in the bracket, so only the call was written.
        self.unsolved_rolls = 0

    @property
    def name(self) -> str:
        return f"zero_cost_collar_{self.call_otm:.0%}"

    def target_shares(self, ctx: BarContext) -> float:
        if not ctx.open_positions and ctx.spot > 0 and ctx.equity > 0:
            self._shares_held = ctx.equity / ctx.spot
        return self._shares_held

    def _solve_put_strike(self, ctx: BarContext, call_premium: float) -> float | None:
        """Bisect on the put's distance out of the money until its premium matches."""
        args = (ctx.years_to_expiry, ctx.rate, ctx.sigma)
        lo, hi = 1e-4, self.max_put_otm

        def premium(otm: float) -> float:
            return bs_price(
                ctx.spot, ctx.spot * (1.0 - otm), *args, "put", q=ctx.div_yield
            )

        # Premium falls as the put moves further out. Need a bracket around the target.
        if premium(lo) < call_premium or premium(hi) > call_premium:
            return None
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            if premium(mid) > call_premium:
                lo = mid
            else:
                hi = mid
            if abs(premium(mid) - call_premium) < self.tolerance:
                break
        return round(ctx.spot * (1.0 - 0.5 * (lo + hi)), 2)

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        if not ctx.is_roll or ctx.expiry is None or ctx.open_positions:
            return []
        call_k = resolve_strike(ctx, "call", "moneyness", 0.0, self.call_otm)
        if call_k is None or self._shares_held <= 0:
            return []

        call_premium = bs_price(
            ctx.spot, call_k, ctx.years_to_expiry, ctx.rate, ctx.sigma, "call",
            q=ctx.div_yield,
        )
        legs = [OptionSpec("call", call_k, ctx.expiry, -self._shares_held)]
        put_k = self._solve_put_strike(ctx, call_premium)
        if put_k is None or put_k <= 0:
            # Write the call alone rather than quietly becoming a different structure.
            self.unsolved_rolls += 1
            return legs
        legs.append(OptionSpec("put", put_k, ctx.expiry, +self._shares_held))
        return legs
