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


# ============ BACKLOG #2-5: holes the round-1 review found, each pinned by a corruption


def _with_split(history, day: str, ratio: float, applied: bool) -> PriceHistory:
    """Inject a split. `applied=False` leaves the pre-split prices unscaled, i.e. the feed
    reported the split but never folded it into history."""
    frame = history.frame.copy()
    total = history.total_return_close.copy()
    frame.loc[pd.Timestamp(day), "Splits"] = ratio
    if not applied:
        pre = frame.index < pd.Timestamp(day)
        for column in ["Open", "High", "Low", "Close"]:
            frame.loc[pre, column] = frame.loc[pre, column] * ratio
        total.loc[pre] = total.loc[pre] * ratio
    return PriceHistory("X", frame, total, "fixture", history.fetched_at)


# ---- #5 splits are now actually cross-referenced, at ANY ratio


@pytest.mark.parametrize("ratio", [1.05, 1.25, 1.3333, 1.40, 1.50, 2.0, 3.0])
def test_unadjusted_split_is_caught_at_every_detectable_ratio(tsla_split, ratio):
    """The hole: split consistency was enforced only by the 40% outlier threshold, so
    anything under e^0.40 = 1.4918 escaped entirely. A 5-for-4, a 4-for-3, a 7-for-5 and
    every 5% stock dividend passed as CLEAN while the report printed the split it had
    failed to check."""
    broken = _with_split(tsla_split, "2022-09-01", ratio, applied=False)
    report = assess(broken, asof=ASOF)
    assert report.unadjusted_splits, f"ratio {ratio} escaped the cross-reference"
    day, seen_ratio, move = report.unadjusted_splits[0]
    assert day == "2022-09-01"
    assert seen_ratio == pytest.approx(ratio)
    assert not report.clean
    assert any("never folded into history" in f for f in report.failures)


@pytest.mark.parametrize("ratio", [1.05, 1.25, 1.50, 3.0])
def test_correctly_applied_split_is_not_flagged(tsla_split, ratio):
    """A split Yahoo already folded in leaves no discontinuity, so it must stay silent."""
    fine = _with_split(tsla_split, "2022-09-01", ratio, applied=True)
    report = assess(fine, asof=ASOF)
    assert report.unadjusted_splits == []
    assert report.clean, report.failures


def test_small_split_escapes_the_outlier_threshold_but_not_the_cross_reference(tsla_split):
    """Proves the two checks are independent, which is the whole point."""
    broken = _with_split(tsla_split, "2022-09-01", 1.25, applied=False)
    report = assess(broken, asof=ASOF)
    assert report.outliers == [], "a 1.25 ratio is only a -22% move, under the threshold"
    assert report.unadjusted_splits, "but the cross-reference must still catch it"


# ---- #3 duplicated rows, and row count vs session count


def test_duplicated_session_row_is_caught(tsla_split):
    """Set differences cannot see a duplicate: `missing` and `unexpected` are both empty."""
    frame = tsla_split.frame.copy()
    dupe = pd.Timestamp("2022-08-24")
    frame = pd.concat([frame, frame.loc[[dupe]]]).sort_index()
    total = pd.concat(
        [tsla_split.total_return_close, tsla_split.total_return_close.loc[[dupe]]]
    ).sort_index()

    report = assess(PriceHistory("TSLA", frame, total, "fixture", tsla_split.fetched_at),
                    asof=ASOF)
    assert dupe in report.duplicate_rows
    assert report.missing_sessions == [] and report.unexpected_rows == []
    assert not report.clean
    assert any("duplicated session row" in f for f in report.failures)


# ---- #2 shortness against the requested window is reported


def test_short_window_is_reported_against_the_request(tsla_split):
    """The live bug: RDDT asked for 5 years, got 2.4, and reported clean. A legitimately
    short history is fine; reporting it as five years is not."""
    report = assess(
        tsla_split,
        requested_start="2021-07-29",
        requested_end="2022-09-30",
        asof=ASOF,
    )
    assert report.missing_head_sessions > 200, report.missing_head_sessions
    assert any("SHORT OF REQUEST" in n for n in report.notes)
    assert "requested window" in report.render()
    # Still not an ERROR: a recent listing is short for a real reason.
    assert report.clean, report.failures


def test_full_window_reports_no_shortfall(tsla_split):
    report = assess(
        tsla_split,
        requested_start=tsla_split.frame.index[0],
        requested_end=tsla_split.frame.index[-1],
        asof=ASOF,
    )
    assert report.missing_head_sessions == 0
    assert report.missing_tail_sessions == 0
    assert not any("SHORT OF REQUEST" in n for n in report.notes)


def test_absent_request_is_disclosed(tsla_split):
    report = assess(tsla_split, asof=ASOF)
    assert any("shortness could not be checked" in n for n in report.notes)


# ---- #4 risk-free coverage is now visible and checkable


def test_thin_risk_free_coverage_is_a_failure(tsla_split):
    """A truncated ^IRX response used to become a five-year constant, a 3.5pp one-signed
    error that alone flips a `sharpe >= 1` verdict."""
    idx = tsla_split.frame.index
    rf = pd.Series(0.0005, index=idx)
    had_print = pd.Series(False, index=idx)
    had_print.iloc[:3] = True  # three real prints for a whole window

    report = assess(tsla_split, rf=rf, rf_had_print=had_print, asof=ASOF)
    assert not report.clean
    assert any("real print on only" in f for f in report.failures)


def test_good_risk_free_coverage_passes(tsla_split):
    idx = tsla_split.frame.index
    report = assess(
        tsla_split,
        rf=pd.Series(0.04, index=idx),
        rf_had_print=pd.Series(True, index=idx),
        asof=ASOF,
    )
    assert report.clean, report.failures
    assert report.rf_sessions_with_print == len(idx)


def test_mis_scaled_risk_free_is_caught(tsla_split):
    """540 read as 5.40 yields a 186% rate, and nothing downstream would question it."""
    idx = tsla_split.frame.index
    rf = pd.Series(0.04, index=idx)
    rf.iloc[10] = 1.856
    report = assess(
        tsla_split, rf=rf, rf_had_print=pd.Series(True, index=idx), asof=ASOF
    )
    assert report.rf_out_of_bounds == 1
    assert not report.clean
    assert any("scaling error" in f for f in report.failures)


def test_split_too_small_to_verify_declines_rather_than_guessing(tsla_split):
    """Honest limit, and where it actually falls.

    A relative tolerance made 1.05 verifiable in both directions, which an absolute log
    tolerance could not do: it flagged correctly-adjusted 1.05 splits as broken. But the
    limit still exists. A 1.01 stock dividend implies a -1% move, indistinguishable from an
    ordinary down day, so the check declines explicitly instead of guessing either way.
    """
    for applied in (True, False):
        broken = _with_split(tsla_split, "2022-09-01", 1.01, applied=applied)
        report = assess(broken, asof=ASOF)
        assert report.unadjusted_splits == []
        assert any("too small to verify" in n for n in report.notes)


def test_relative_tolerance_is_what_makes_small_splits_verifiable(tsla_split):
    """Pin the reason. Under the old absolute 0.15 log tolerance, the band around
    -log(1.05) = -0.0488 spanned -0.199 to +0.101 and swallowed almost any trading day."""
    import numpy as np
    from canalyst.data.quality import SPLIT_MATCH_TOLERANCE

    expected = -np.log(1.05)
    relative_band = SPLIT_MATCH_TOLERANCE * abs(expected)
    assert relative_band < 0.02, "the band must be narrow enough to exclude a normal day"
    # A correctly-adjusted 1.05 split sits outside it; an unadjusted one sits inside.
    applied = _with_split(tsla_split, "2022-09-01", 1.05, applied=True)
    unapplied = _with_split(tsla_split, "2022-09-01", 1.05, applied=False)
    assert assess(applied, asof=ASOF).unadjusted_splits == []
    assert assess(unapplied, asof=ASOF).unadjusted_splits


def test_reverse_split_is_also_checked(tsla_split):
    """A 1-for-3 reverse split is ratio 0.3333 and must not slip past the floor check."""
    frame = tsla_split.frame.copy()
    total = tsla_split.total_return_close.copy()
    day = pd.Timestamp("2022-09-01")
    frame.loc[day, "Splits"] = 1 / 3
    pre = frame.index < day
    for column in ["Open", "High", "Low", "Close"]:
        frame.loc[pre, column] = frame.loc[pre, column] / 3.0
    total.loc[pre] = total.loc[pre] / 3.0
    report = assess(
        PriceHistory("X", frame, total, "fixture", tsla_split.fetched_at), asof=ASOF
    )
    assert report.unadjusted_splits, "a reverse split left unadjusted must be caught"
    assert not any("too small to verify" in n for n in report.notes)
