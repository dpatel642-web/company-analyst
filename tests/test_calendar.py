"""Sessions and option expiries.

The load-bearing case is 2025-04-18, which was simultaneously April's third Friday and
Good Friday. Any implementation that assumes the third Friday is always a trading day
gets that month's expiry wrong, and then prices an option against a closed market.
"""

from __future__ import annotations

import pandas as pd
import pytest

from canalyst.data.calendar import (
    align_to_session,
    last_completed_session,
    monthly_expiries,
    roll_schedule,
    sessions,
    third_fridays,
)

WINDOW = ("2021-07-29", "2026-07-29")


def test_session_count_matches_nyse():
    """1255 sessions over this five-year window, verified against exchange_calendars."""
    s = sessions(*WINDOW)
    assert len(s) == 1255
    assert s[0] == pd.Timestamp("2021-07-29")
    assert s[-1] == pd.Timestamp("2026-07-29")


def test_sessions_exclude_known_holidays():
    s = sessions("2025-01-01", "2025-12-31")
    for holiday in ["2025-01-01", "2025-07-04", "2025-12-25", "2025-04-18"]:
        assert pd.Timestamp(holiday) not in s, holiday


def test_sessions_are_naive_and_midnight():
    s = sessions("2025-01-01", "2025-03-31")
    assert s.tz is None
    assert (s.normalize() == s).all()


# ------------------------------------------------------------------ the Good Friday case


def test_good_friday_is_aprils_third_friday_2025():
    """Premise check. If this ever fails the regression below is testing nothing."""
    fridays = [d for d in third_fridays("2025-04-01", "2025-04-30")
               if d.year == 2025 and d.month == 4]
    assert fridays == [pd.Timestamp("2025-04-18")]
    assert pd.Timestamp("2025-04-18") not in sessions("2025-04-01", "2025-04-30")


def test_holiday_expiry_shifts_to_preceding_session():
    s = sessions(*WINDOW)
    april = [d for d in monthly_expiries(s) if d.year == 2025 and d.month == 4]
    assert april == [pd.Timestamp("2025-04-17")]


def test_expiries_are_all_real_sessions():
    s = sessions(*WINDOW)
    expiries = monthly_expiries(s)
    assert len(expiries) == 60
    assert expiries.isin(s).all()


# --------------------------------------------------------------------------- alignment


def test_align_snaps_backwards_never_forwards():
    """Rolling forward would use a price the trader had not yet seen."""
    s = sessions("2025-04-01", "2025-04-30")
    aligned = align_to_session(pd.DatetimeIndex([pd.Timestamp("2025-04-18")]), s)
    assert aligned[0] == pd.Timestamp("2025-04-17")
    assert aligned[0] < pd.Timestamp("2025-04-18")


def test_align_passes_through_real_sessions():
    s = sessions("2025-03-01", "2025-03-31")
    target = pd.Timestamp("2025-03-21")
    assert align_to_session(pd.DatetimeIndex([target]), s)[0] == target


def test_align_drops_targets_before_all_sessions():
    s = sessions("2025-03-01", "2025-03-31")
    assert len(align_to_session(pd.DatetimeIndex([pd.Timestamp("2020-01-06")]), s)) == 0


def test_align_deduplicates():
    """Two targets landing on one session must not produce a duplicate roll."""
    s = sessions("2025-04-01", "2025-04-30")
    targets = pd.DatetimeIndex([pd.Timestamp("2025-04-18"), pd.Timestamp("2025-04-19")])
    assert len(align_to_session(targets, s)) == 1


# ----------------------------------------------------------------------- roll schedule


def test_roll_schedule_is_consecutive_expiries():
    s = sessions(*WINDOW)
    sched = roll_schedule(s)
    expiries = monthly_expiries(s)
    assert len(sched) == len(expiries) - 1
    assert list(sched.index) == list(expiries[:-1])
    assert list(sched["expiry"]) == list(expiries[1:])


def test_roll_always_precedes_its_expiry():
    sched = roll_schedule(sessions(*WINDOW))
    assert (sched["expiry"].values > sched.index.values).all()


def test_roll_gaps_are_monthly():
    sched = roll_schedule(sessions(*WINDOW))
    gaps = (sched["expiry"].values - sched.index.values).astype("timedelta64[D]").astype(int)
    assert gaps.min() >= 25 and gaps.max() <= 40


def test_roll_schedule_handles_too_short_a_window():
    sched = roll_schedule(sessions("2025-03-01", "2025-03-31"))
    assert len(sched) == 0


# ---------------------------------------------------------------------------- freshness


def test_in_progress_session_is_not_completed():
    """2026-07-29 14:00 UTC is 10:00 ET, mid-session. The bar is still forming."""
    asof = pd.Timestamp("2026-07-29T14:00:00", tz="UTC")
    assert last_completed_session(asof) == pd.Timestamp("2026-07-28")


def test_session_counts_once_the_bell_has_rung():
    """21:00 UTC is 17:00 ET, after the 16:00 close."""
    asof = pd.Timestamp("2026-07-29T21:00:00", tz="UTC")
    assert last_completed_session(asof) == pd.Timestamp("2026-07-29")


def test_weekend_falls_back_to_friday():
    asof = pd.Timestamp("2026-08-02T12:00:00", tz="UTC")  # a Sunday
    assert last_completed_session(asof) == pd.Timestamp("2026-07-31")


def test_naive_asof_is_treated_as_utc():
    assert last_completed_session(pd.Timestamp("2026-07-29T14:00:00")) == pd.Timestamp(
        "2026-07-28"
    )
