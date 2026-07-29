"""Buy and hold. The benchmark every overlay is measured against.

IT MUST USE THE SAME REBALANCE RULE AS THE OVERLAYS, and this is not a detail.

A constant one-share benchmark leaves dividends and interest sitting in cash, while the
collateralised overlays rebalance `shares = equity/spot` and therefore compound that cash
back into stock. Comparing them then measures two different dividend-reinvestment
policies and attributes the whole difference to the option overlay.

Measured on a PG-like path (5y, 2.4% yield): the constant-share benchmark returns
+175.99% and ends with 23.2% of its capital idle in cash, while the same benchmark under
the overlays' own rebalance rule returns +183.25%. The convention mismatch alone was
worth 7.26 percentage points, all of it flattering the overlay.

Sharper still: at zero pricing vol a written call is worth exactly intrinsic, so a
covered call is economically identical to buy-and-hold, which is what the zero-vol
collapse test exists to prove. Add a dividend and the collapse breaks by 12.3% of initial
capital, in the overlay's favour, while the accounting identity passes. At that point the
test is measuring dividend policy and nothing about options at all.

So `fully_invested` defaults to True: hold `equity/spot` shares, which is identically one
share when there is no dividend and no interest, and is the honest comparator when there
is.
"""

from __future__ import annotations

from .base import BarContext, OptionSpec


class BuyHold:
    name = "buy_and_hold"

    def __init__(self, shares: float = 1.0, fully_invested: bool = True) -> None:
        self.shares = float(shares)
        self.fully_invested = bool(fully_invested)
        self._shares_held = float(shares)

    def target_shares(self, ctx: BarContext) -> float:
        if not self.fully_invested:
            return self.shares
        # No options are ever open, so every bar is a legal rebalance point. Sweeping
        # cash into stock each bar is what "fully invested" means, and it matches the
        # policy the overlays follow whenever their book is empty.
        if ctx.spot > 0 and ctx.equity > 0:
            self._shares_held = ctx.equity / ctx.spot
        return self._shares_held

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        return []
