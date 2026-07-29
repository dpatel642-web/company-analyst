"""What a strategy is allowed to say, and what the engine decides on its own.

A strategy answers two narrow questions per bar: how many shares it wants to hold, and
what new option positions it wants to open. Everything else, settlement at expiry,
dividend receipt, interest on cash, and marking every open position daily, belongs to
the engine.

That split is deliberate. The handout's bugs both live in code where strategy logic and
ledger arithmetic were interleaved, so an expiring position's payoff could be dropped by
a later assignment to the same cell. A strategy here cannot touch the ledger, so it
cannot lose money out of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..options.bs import Kind


@dataclass(frozen=True)
class OptionSpec:
    """A position to open. Negative quantity is short."""

    kind: Kind
    strike: float
    expiry: pd.Timestamp
    quantity: float

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike!r}")
        if self.quantity == 0:
            raise ValueError("quantity of zero is not a position")


@dataclass(frozen=True)
class BarContext:
    """Everything a strategy may look at on a given bar.

    Deliberately contains no future data. `expiry` is the expiry of the contract to be
    written *today*, which the schedule fixes in advance, so knowing it is not
    lookahead. Prices and vol are as of today's close.
    """

    date: pd.Timestamp
    spot: float
    sigma: float
    rate: float
    dividend: float
    is_roll: bool
    expiry: pd.Timestamp | None
    years_to_expiry: float
    open_positions: tuple["OpenPosition", ...]
    #: Capital available right now, after marking and settling but before trading.
    #: Strategies size against this so a losing run deleverages instead of borrowing.
    equity: float = 0.0
    #: Continuous dividend yield, used for pricing AND for strike selection. Both matter:
    #: with q omitted the premium is wrong, and the strike chosen for a nominal 0.25 delta
    #: actually sits at 0.2375, so the pre-specified parameter quietly stops being the
    #: parameter in force.
    div_yield: float = 0.0


@dataclass(frozen=True)
class OpenPosition:
    """An option the book is currently carrying, with the mark it last received."""

    spec: OptionSpec
    opened_on: pd.Timestamp
    last_mark: float

    @property
    def kind(self) -> Kind:
        return self.spec.kind

    @property
    def strike(self) -> float:
        return self.spec.strike

    @property
    def expiry(self) -> pd.Timestamp:
        return self.spec.expiry

    @property
    def quantity(self) -> float:
        return self.spec.quantity


class Strategy(Protocol):
    name: str

    def target_shares(self, ctx: BarContext) -> float:
        """Shares to hold at the close of this bar."""
        ...

    def options_to_open(self, ctx: BarContext) -> list[OptionSpec]:
        """New positions to open at this bar's close. Empty on non-roll days."""
        ...
