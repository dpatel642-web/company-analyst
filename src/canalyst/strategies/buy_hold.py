"""Buy and hold one share. The benchmark every overlay is measured against."""

from __future__ import annotations

from .base import BarContext, OptionSpec


class BuyHold:
    name = "buy_and_hold"

    def __init__(self, shares: float = 1.0) -> None:
        self.shares = float(shares)

    def target_shares(self, ctx: BarContext) -> float:
        return self.shares

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        return []
