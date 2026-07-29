"""Offline fixtures. No test in this suite may touch the network.

A test suite that needs Yahoo to be up is a test suite that goes red for reasons
unrelated to the code, and then gets ignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from canalyst.data.prices import PriceHistory

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str, ticker: str, source: str = "fixture") -> PriceHistory:
    frame = pd.read_csv(FIXTURES / name, index_col="Date", parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    total_return = frame.pop("TotalReturnClose")
    return PriceHistory(
        ticker=ticker,
        frame=frame,
        total_return_close=total_return,
        source=source,
        fetched_at=pd.Timestamp("2026-07-29T16:00:00"),
    )


@pytest.fixture
def tsla_split() -> PriceHistory:
    """Aug-Sep 2022, spanning the 3-for-1 split on 2022-08-25."""
    return load_fixture("TSLA_2022_split.csv", "TSLA")


@pytest.fixture
def pg_dividends() -> PriceHistory:
    """H1 2024 Procter & Gamble, with two ex-dividend dates."""
    return load_fixture("PG_2024_dividends.csv", "PG")


def mutate(
    history: PriceHistory, edits: dict[tuple[str, str], float]
) -> PriceHistory:
    """Return a copy with `{(column, date): value}` edits applied.

    A dict rather than kwargs because the keys are (column, date) tuples and Python
    keyword names must be strings.
    """
    frame = history.frame.copy()
    total = history.total_return_close.copy()
    for (column, day), value in edits.items():
        frame.loc[pd.Timestamp(day), column] = value
    return PriceHistory(
        ticker=history.ticker,
        frame=frame,
        total_return_close=total,
        source=history.source,
        fetched_at=history.fetched_at,
    )
