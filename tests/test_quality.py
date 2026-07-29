"""Integrity checks, tested by breaking good data on purpose.

A check that only ever runs against clean data proves nothing. Every check here gets
two tests: it stays quiet on real data, and it fires on a specific corruption. The
corruptions are modelled on the failure modes that actually occur:

  - a provider silently drops a session
  - a split is applied to the price series but not the total-return series
    (this is the 3-for-1 regression: it surfaces as a 3x jump)
  - an in-progress bar survives into the series
  - the second source disagrees about a recent close
"""

from __future__ import annotations

import pandas as pd
import pytest

from canalyst.data.prices import PriceHistory, drop_incomplete_bar
from canalyst.data.quality import OUTLIER_LOG_RETURN, assess

from .conftest import mutate

ASOF = pd.Timestamp("2026-07-29T14:00:00", tz="UTC")


def _drop_day(history: PriceHistory, day: str) -> PriceHistory:
    keep = history.frame.index != pd.Timestamp(day)
    return PriceHistory(
        ticker=history.ticker,
        frame=history.frame.loc[keep].copy(),
        total_return_close=history.total_return_close.loc[keep].copy(),
        source=history.source,
        fetched_at=history.fetched_at,
    )


# ------------------------------------------------------------------- baseline is clean


def test_real_tsla_window_is_clean(tsla_split):
    report = assess(tsla_split, asof=ASOF)
    assert report.clean, report.failures
    assert report.rows == report.expected_sessions
    assert report.missing_sessions == []


def test_real_dividend_payer_is_clean(pg_dividends):
    """Exercises the dividend branch of the adjustment check, not just the zero case."""
    report = assess(pg_dividends, asof=ASOF)
    assert report.clean, report.failures
    assert report.dividend_total > 0
    assert report.adjustment_consistent is True


def test_split_is_reported_without_being_an_error(tsla_split):
    """The 3-for-1 is present in the actions feed and must NOT look like a fault."""
    report = assess(tsla_split, asof=ASOF)
    assert report.splits == {"2022-08-25": 3.0}
    assert report.outliers == []
    assert report.clean


# ------------------------------------------------------------------------ completeness


def test_missing_session_is_caught(tsla_split):
    broken = _drop_day(tsla_split, "2022-08-24")
    report = assess(broken, asof=ASOF)
    assert not report.clean
    assert pd.Timestamp("2022-08-24") in report.missing_sessions
    assert any("missing trading session" in f for f in report.failures)


def test_row_that_is_not_a_session_is_caught(tsla_split):
    frame = tsla_split.frame.copy()
    saturday = pd.Timestamp("2022-08-27")  # a Saturday
    frame.loc[saturday] = frame.iloc[-1]
    frame = frame.sort_index()
    total = tsla_split.total_return_close.copy()
    total.loc[saturday] = total.iloc[-1]
    report = assess(
        PriceHistory("TSLA", frame, total.sort_index(), "fixture", tsla_split.fetched_at),
        asof=ASOF,
    )
    assert not report.clean
    assert saturday in report.unexpected_rows


# -------------------------------------------------------------------- impossible bars


@pytest.mark.parametrize(
    "column,value,label",
    [
        ("Low", 9_999.0, "high < low"),
        ("Close", 9_999.0, "close outside [low, high]"),
        ("Open", 0.01, "open outside [low, high]"),
        ("Volume", 0.0, "non-positive volume"),
    ],
)
def test_impossible_bar_is_caught(tsla_split, column, value, label):
    broken = mutate(tsla_split, {(column, "2022-08-24"): value})
    report = assess(broken, asof=ASOF)
    assert not report.clean
    assert report.bad_bars[label] >= 1, report.bad_bars


def test_nan_price_is_caught(tsla_split):
    broken = mutate(tsla_split, {("Close", "2022-08-24"): float("nan")})
    report = assess(broken, asof=ASOF)
    assert not report.clean
    assert report.bad_bars["missing price"] >= 1


# ------------------------------------------- the split regression: a 3x discontinuity


def test_unapplied_split_is_caught(tsla_split):
    """Simulate the split NOT having been folded into history before 2022-08-25.

    Yahoo adjusts retroactively, so pre-split rows should already be divided by three.
    Multiplying them back by three reproduces exactly what an unadjusted or
    half-adjusted feed looks like, and the check must catch it.
    """
    frame = tsla_split.frame.copy()
    pre = frame.index < pd.Timestamp("2022-08-25")
    for column in ["Open", "High", "Low", "Close"]:
        frame.loc[pre, column] = frame.loc[pre, column] * 3.0
    broken = PriceHistory(
        "TSLA", frame, tsla_split.total_return_close.copy(), "fixture",
        tsla_split.fetched_at,
    )
    report = assess(broken, asof=ASOF)
    assert not report.clean
    assert report.unexplained_outliers, "a 3x price gap must be flagged"
    day, move = report.unexplained_outliers[0]
    assert day == pd.Timestamp("2022-08-25")
    assert move < -0.60  # a 3-for-1 shows up as roughly -67%


def test_outlier_threshold_is_actually_enforced(tsla_split):
    """A move just over the threshold fires; the clean series has none."""
    frame = tsla_split.frame.copy()
    target = pd.Timestamp("2022-08-24")
    for column in ["Open", "High", "Low", "Close"]:
        frame.loc[target, column] = frame.loc[target, column] * 2.0
    report = assess(
        PriceHistory("TSLA", frame, tsla_split.total_return_close.copy(), "fixture",
                     tsla_split.fetched_at),
        asof=ASOF,
    )
    assert any(abs(m) > OUTLIER_LOG_RETURN * 0.9 for _, m in report.outliers)
    assert not report.clean


def test_inconsistent_adjustment_between_series_is_caught(tsla_split):
    """Split applied to the price series but not the total-return series."""
    total = tsla_split.total_return_close.copy()
    pre = total.index < pd.Timestamp("2022-08-25")
    total.loc[pre] = total.loc[pre] * 3.0
    broken = PriceHistory(
        "TSLA", tsla_split.frame.copy(), total, "fixture", tsla_split.fetched_at
    )
    report = assess(broken, asof=ASOF)
    assert report.adjustment_consistent is False
    assert not report.clean


# ----------------------------------------------------------------------- partial bars


def test_bar_from_an_unclosed_session_is_caught():
    """A row dated today, mid-session, must be flagged rather than silently trusted."""
    idx = pd.DatetimeIndex(["2026-07-27", "2026-07-28", "2026-07-29"])
    frame = pd.DataFrame(
        {
            "Open": [300.0, 305.0, 306.0],
            "High": [310.0, 312.0, 311.0],
            "Low": [299.0, 300.0, 300.0],
            "Close": [305.0, 307.0, 308.0],
            "Volume": [1e7, 1e7, 1e7],
            "Dividends": [0.0, 0.0, 0.0],
            "Splits": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    history = PriceHistory(
        "TSLA", frame, frame["Close"].copy(), "fixture",
        pd.Timestamp("2026-07-29T14:00:00"),
    )
    report = assess(history, asof=ASOF)
    assert not report.clean
    assert report.bad_bars.get("bar for a session that has not closed") == 1


def test_drop_incomplete_bar_removes_it():
    idx = pd.DatetimeIndex(["2026-07-27", "2026-07-28", "2026-07-29"])
    frame = pd.DataFrame(
        {c: [1.0, 1.0, 1.0] for c in
         ["Open", "High", "Low", "Close", "Volume", "Dividends", "Splits"]},
        index=idx,
    )
    history = PriceHistory("TSLA", frame, frame["Close"].copy(), "fixture",
                           pd.Timestamp("2026-07-29T14:00:00"))
    trimmed = drop_incomplete_bar(history, asof=ASOF)
    assert trimmed.frame.index[-1] == pd.Timestamp("2026-07-28")
    assert len(trimmed.frame) == 2
    # Idempotent: dropping again changes nothing.
    assert len(drop_incomplete_bar(trimmed, asof=ASOF).frame) == 2


def test_staleness_is_noted_not_silently_accepted(tsla_split):
    report = assess(tsla_split, asof=ASOF)
    assert any("behind" in note for note in report.notes)


# ------------------------------------------------------------------ second-source tail


def test_agreeing_second_source_passes(tsla_split):
    tail = tsla_split.close.tail(4).copy()
    report = assess(tsla_split, tail_closes=tail, tail_source="nasdaq", asof=ASOF)
    assert report.tail_checked == 4
    assert report.tail_disagreements == []
    assert report.clean


def test_disagreeing_second_source_is_caught(tsla_split):
    tail = tsla_split.close.tail(4).copy()
    tail.iloc[-1] = tail.iloc[-1] * 1.05  # 5% off, well past the 0.5% tolerance
    report = assess(tsla_split, tail_closes=tail, tail_source="nasdaq", asof=ASOF)
    assert not report.clean
    assert len(report.tail_disagreements) == 1
    assert any("disagree" in f for f in report.failures)


def test_tiny_second_source_difference_is_tolerated(tsla_split):
    """Rounding between venues is not a data error."""
    tail = tsla_split.close.tail(4) * 1.001
    report = assess(tsla_split, tail_closes=tail, tail_source="nasdaq", asof=ASOF)
    assert report.tail_disagreements == []
    assert report.clean


def test_absent_second_source_is_disclosed_not_hidden(tsla_split):
    report = assess(tsla_split, asof=ASOF)
    assert any("no independent source" in note for note in report.notes)
    assert report.clean  # a missing second source is a caveat, not a failure


# ------------------------------------------------------------------------- disclosures


def test_report_always_discloses_its_two_known_limits(tsla_split):
    report = assess(tsla_split, asof=ASOF)
    joined = " ".join(report.notes)
    assert "split-adjusted" in joined
    assert "full-window independence is NOT verified" in joined


def test_render_shows_verdict_and_reasons(tsla_split):
    clean = assess(tsla_split, asof=ASOF).render()
    assert "CLEAN" in clean

    broken = assess(_drop_day(tsla_split, "2022-08-24"), asof=ASOF).render()
    assert "FAILED" in broken
    assert "missing trading session" in broken


def test_assess_rejects_empty_history(tsla_split):
    empty = PriceHistory(
        "TSLA", tsla_split.frame.iloc[0:0], tsla_split.total_return_close.iloc[0:0],
        "fixture", tsla_split.fetched_at,
    )
    with pytest.raises(ValueError, match="empty"):
        assess(empty, asof=ASOF)
