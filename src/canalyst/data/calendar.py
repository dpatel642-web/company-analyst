"""Trading sessions and monthly option expiries, from the authoritative NYSE calendar.

Why a calendar library instead of `pd.bdate_range`: business days are not sessions.
Without the real holiday list you cannot distinguish "the market was closed" from
"this row is missing", so a series with holes reads as complete. That distinction is
the whole basis of the completeness check in `quality.py`.

Monthly equity options expire on the third Friday. When that Friday is a holiday the
expiry moves *earlier*, to the preceding session. This is not hypothetical inside our
window: 2025-04-18 was both the third Friday of April and Good Friday, so that month's
expiry was Thursday 2025-04-17.
"""

from __future__ import annotations

import datetime as dt

import exchange_calendars as xcals
import pandas as pd

DEFAULT_CALENDAR = "XNYS"


def _naive(index) -> pd.DatetimeIndex:
    """Drop timezone and time-of-day so every date in the project compares equal."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def sessions(
    start: dt.date | str | pd.Timestamp,
    end: dt.date | str | pd.Timestamp,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.DatetimeIndex:
    """Every trading session in [start, end], inclusive."""
    cal = xcals.get_calendar(calendar)
    return _naive(cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end)))


def last_completed_session(
    asof: dt.datetime | pd.Timestamp | None = None,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.Timestamp:
    """The most recent session whose closing bell has actually rung.

    If `asof` falls inside a session, that session is *excluded*: its bar is still
    forming. Treating an in-progress bar as a close silently corrupts the final day
    of every series, which is the single easiest way to publish a wrong number.
    """
    cal = xcals.get_calendar(calendar)
    now = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz="UTC")
    if now.tz is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    window = cal.sessions_in_range(
        (now - pd.Timedelta(days=30)).tz_convert(None).normalize(),
        (now + pd.Timedelta(days=1)).tz_convert(None).normalize(),
    )
    for session in reversed(list(window)):
        close = cal.session_close(session)
        if close <= now:
            return _naive([session])[0]
    raise RuntimeError("no completed session found in the trailing 30 days")


def third_fridays(
    start: dt.date | str | pd.Timestamp,
    end: dt.date | str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Calendar third Fridays, ignoring holidays. Padded a month either side."""
    lo = pd.Timestamp(start).normalize() - pd.offsets.MonthBegin(1)
    hi = pd.Timestamp(end).normalize() + pd.offsets.MonthEnd(1)
    return pd.DatetimeIndex(pd.date_range(lo, hi, freq="WOM-3FRI"))


def align_to_session(
    targets: pd.DatetimeIndex,
    trading_sessions: pd.DatetimeIndex,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.DatetimeIndex:
    """Snap each target date back to the last session on or before it.

    Backwards, never forwards: a holiday expiry settles on the preceding session,
    and rolling *after* the intended date would peek at prices the trader had not
    seen yet.

    Targets outside the session range are dropped at BOTH ends. Clamping a target that
    lies beyond the last session would invent an expiry on whatever the final bar
    happens to be: aligning April's third Friday against a series ending 2025-03-31
    once produced a "monthly expiry" on a Monday, which then drove a fabricated final
    roll. Out of range means no expiry, not the nearest one.
    """
    sessions_sorted = pd.DatetimeIndex(trading_sessions).sort_values()
    if len(sessions_sorted) == 0:
        return pd.DatetimeIndex([])
    out: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for target in _naive(targets):
        # Bound on the SNAPPED session, not the raw target. Testing the unsnapped third
        # Friday discards a holiday-shifted expiry that is genuinely present: with data
        # ending 2025-04-17, the 04-18 Good Friday target reads as out of range even
        # though 04-17 IS that month's expiry and IS the last bar. All six
        # holiday-shifted expiries in 2008-2026 failed this way.
        loc = sessions_sorted.searchsorted(target, side="right")
        if loc == 0:
            continue  # target predates all known sessions
        session = sessions_sorted[loc - 1]
        # A target beyond the data snaps to the final bar, which would invent an expiry
        # on whatever that bar happens to be. Only keep it when the target is genuinely
        # within reach of a real session: either the target IS a session, or the snapped
        # session is the calendar-correct predecessor (a holiday shift), which means the
        # gap between them contains no sessions at all.
        if target > sessions_sorted[-1] and target not in sessions_sorted:
            gap = sessions(session, target, calendar=calendar)
            if len(gap) > 1 or (target - session).days > 4:
                continue
        if session not in seen:
            out.append(session)
            seen.add(session)
    return pd.DatetimeIndex(out)


def monthly_expiries(
    trading_sessions: pd.DatetimeIndex,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.DatetimeIndex:
    """Monthly option expiries that exist as sessions within the given series."""
    sessions_sorted = pd.DatetimeIndex(trading_sessions).sort_values()
    if len(sessions_sorted) == 0:
        return pd.DatetimeIndex([])
    fridays = third_fridays(sessions_sorted[0], sessions_sorted[-1])
    aligned = align_to_session(fridays, sessions_sorted, calendar=calendar)
    return aligned[(aligned >= sessions_sorted[0]) & (aligned <= sessions_sorted[-1])]


def roll_schedule(trading_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Consecutive (roll, expiry) pairs: write on one expiry, expire on the next.

    Returns a frame indexed by roll date with an `expiry` column. The final expiry
    has no successor inside the data and is therefore not a roll date.
    """
    expiries = monthly_expiries(trading_sessions)
    if len(expiries) < 2:
        return pd.DataFrame({"expiry": pd.DatetimeIndex([])}, index=pd.DatetimeIndex([]))
    return pd.DataFrame({"expiry": expiries[1:]}, index=expiries[:-1]).rename_axis("roll")
