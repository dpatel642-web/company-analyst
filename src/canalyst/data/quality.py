"""Data integrity checks. Nothing downstream is reportable without a clean report.

Honest statement of what these checks can and cannot do.

CAN detect: missing sessions, impossible bars, a split that was applied inconsistently,
a stale or in-progress final bar, a wrong recent close.

CANNOT detect: a uniformly wrong five-year history. Internal consistency checks pass
happily on a series that is smoothly, systematically wrong, and catching that needs a
genuinely independent source for the whole window. There is no longer a free no-key one
(Stooq is bot-gated as of 2026-07-29, Nasdaq serves only ~10 sessions). So the recent
tail is corroborated externally and the body is corroborated only internally. That gap
is stated in the report rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import last_completed_session, sessions
from .prices import PriceHistory

#: A daily move this large is either a real event or a broken adjustment. Every
#: breach must be explained individually, never waved through in aggregate.
OUTLIER_LOG_RETURN = 0.40
#: Tolerance for the independent close comparison on the recent tail.
TAIL_TOLERANCE = 0.005
#: How close a return on a split date must be to `-log(ratio)` before it counts as
#: evidence the split was never folded in, expressed as a FRACTION of `log(ratio)`.
#:
#: Relative, not absolute. An absolute tolerance in log space is meaningless across
#: ratios: 0.15 around `-log(1.05) = -0.049` spans -0.199 to +0.101, which swallows almost
#: any ordinary trading day and flags correctly-adjusted small splits as broken.
SPLIT_MATCH_TOLERANCE = 0.25
#: Below this ratio the expected discontinuity is inside normal daily noise and cannot be
#: separated from it, so the check declines and says so rather than guessing. A 1% stock
#: dividend simply is not distinguishable from a 1% down day by this method.
SPLIT_MIN_DETECTABLE_RATIO = 1.03
#: A shortfall against the requested window larger than this many sessions is reported
#: prominently. It is not an error (a recent listing is legitimately short) but it must
#: never be invisible, because a 2-year series in a table headed "5y" is a wrong answer.
SHORTFALL_SESSIONS = 5


@dataclass
class DataQualityReport:
    ticker: str
    source: str
    fetched_at: pd.Timestamp
    first_session: pd.Timestamp
    last_session: pd.Timestamp
    rows: int
    expected_sessions: int
    missing_sessions: list[pd.Timestamp] = field(default_factory=list)
    unexpected_rows: list[pd.Timestamp] = field(default_factory=list)
    duplicate_rows: list[pd.Timestamp] = field(default_factory=list)
    bad_bars: dict[str, int] = field(default_factory=dict)
    splits: dict[str, float] = field(default_factory=dict)
    #: Splits whose date carries a return shaped like the split ratio, meaning the feed
    #: never folded the split into history. Distinct from `outliers`: a small-ratio split
    #: never breaches the 40% threshold and so is invisible to that check.
    unadjusted_splits: list[tuple[str, float, float]] = field(default_factory=list)
    #: What was asked for, so shortness can be seen rather than inferred.
    requested_start: pd.Timestamp | None = None
    requested_end: pd.Timestamp | None = None
    missing_head_sessions: int = 0
    missing_tail_sessions: int = 0
    #: Risk-free coverage, the one series that previously had no integrity checking at all.
    rf_sessions_with_print: int | None = None
    rf_sessions_total: int | None = None
    rf_out_of_bounds: int = 0
    dividend_total: float = 0.0
    outliers: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    unexplained_outliers: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    adjustment_consistent: bool | None = None
    tail_checked: int = 0
    tail_disagreements: list[tuple[pd.Timestamp, float, float]] = field(
        default_factory=list
    )
    tail_source: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        """Hard problems. A non-empty list means do not read the performance numbers."""
        out: list[str] = []
        if self.missing_sessions:
            out.append(
                f"{len(self.missing_sessions)} missing trading session(s), "
                f"first {self.missing_sessions[0].date()}"
            )
        if self.unexpected_rows:
            out.append(
                f"{len(self.unexpected_rows)} row(s) that are not NYSE sessions, "
                f"first {self.unexpected_rows[0].date()}"
            )
        if self.duplicate_rows:
            # Set differences cannot see a duplicate, so this needs its own check. A
            # duplicated bar yields a spurious 0% return and double-counts the real one.
            out.append(
                f"{len(self.duplicate_rows)} duplicated session row(s), "
                f"first {self.duplicate_rows[0].date()}"
            )
        if self.rows != self.expected_sessions and not (
            self.missing_sessions or self.unexpected_rows or self.duplicate_rows
        ):
            out.append(
                f"row count {self.rows} does not match {self.expected_sessions} "
                "sessions, and no missing, extra or duplicated row explains it"
            )
        for day, ratio, move in self.unadjusted_splits:
            out.append(
                f"split {ratio:g}-for-1 on {day} coincides with a {move:+.1%} move, "
                "which is the shape of a split that was never folded into history"
            )
        if self.rf_sessions_total and self.rf_sessions_with_print is not None:
            covered = self.rf_sessions_with_print / self.rf_sessions_total
            if covered < 0.80:
                out.append(
                    f"risk-free rate has a real print on only "
                    f"{self.rf_sessions_with_print} of {self.rf_sessions_total} "
                    f"sessions ({covered:.0%}); the rest is carried forward"
                )
        if self.rf_out_of_bounds:
            out.append(
                f"{self.rf_out_of_bounds} risk-free observation(s) outside a plausible "
                "range, which is the signature of a scaling error"
            )
        for label, count in self.bad_bars.items():
            if count:
                out.append(f"{count} bar(s) failed check: {label}")
        if self.unexplained_outliers:
            day, move = self.unexplained_outliers[0]
            out.append(
                f"{len(self.unexplained_outliers)} unexplained move(s) over "
                f"{OUTLIER_LOG_RETURN:.0%}, first {day.date()} at {move:+.1%}"
            )
        if self.adjustment_consistent is False:
            out.append("split/dividend adjustment is internally inconsistent")
        if self.tail_disagreements:
            day, ours, theirs = self.tail_disagreements[0]
            out.append(
                f"{len(self.tail_disagreements)} recent close(s) disagree with "
                f"{self.tail_source}, first {day.date()}: {ours:.2f} vs {theirs:.2f}"
            )
        return out

    @property
    def clean(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [
            "Data quality",
            "------------",
            f"  ticker            {self.ticker}",
            f"  source            {self.source}",
            f"  fetched           {self.fetched_at:%Y-%m-%d %H:%M} UTC",
            f"  window            {self.first_session:%Y-%m-%d} .. "
            f"{self.last_session:%Y-%m-%d}",
            f"  rows / sessions   {self.rows} / {self.expected_sessions}",
            f"  missing sessions  {len(self.missing_sessions)}",
            f"  impossible bars   {sum(self.bad_bars.values())}",
            f"  splits in window  {self.splits or 'none'}",
            f"  duplicated rows   {len(self.duplicate_rows)}",
            f"  dividends paid    {self.dividend_total:.4f}",
            f"  moves over {OUTLIER_LOG_RETURN:.0%}     "
            f"{len(self.outliers)} ({len(self.unexplained_outliers)} unexplained)",
            f"  adjustment self-consistent  {self.adjustment_consistent}",
        ]
        if self.requested_start is not None or self.requested_end is not None:
            def _fmt(stamp: pd.Timestamp | None) -> str:
                return f"{stamp:%Y-%m-%d}" if stamp is not None else "(unset)"

            lines.append(
                f"  requested window  {_fmt(self.requested_start)} .. "
                f"{_fmt(self.requested_end)}"
            )
            lines.append(
                f"  shortfall         {self.missing_head_sessions} session(s) at the "
                f"start, {self.missing_tail_sessions} at the end"
            )
        if self.rf_sessions_total:
            printed = (
                f"{self.rf_sessions_with_print}"
                if self.rf_sessions_with_print is not None
                else "unknown"
            )
            lines.append(
                f"  risk-free prints  {printed} of {self.rf_sessions_total}"
                + (f", {self.rf_out_of_bounds} out of bounds" if self.rf_out_of_bounds else "")
            )
        if self.tail_source:
            lines.append(
                f"  recent tail vs {self.tail_source}   "
                f"{self.tail_checked} session(s) compared, "
                f"{len(self.tail_disagreements)} disagreement(s)"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        verdict = "CLEAN" if self.clean else "FAILED"
        lines.append(f"  verdict           {verdict}")
        for failure in self.failures:
            lines.append(f"    - {failure}")
        return "\n".join(lines)


def _check_bars(frame: pd.DataFrame) -> dict[str, int]:
    high, low = frame["High"], frame["Low"]
    close, open_ = frame["Close"], frame["Open"]
    eps = 1e-9
    return {
        "high < low": int((high < low - eps).sum()),
        "close outside [low, high]": int(
            ((close > high + eps) | (close < low - eps)).sum()
        ),
        "open outside [low, high]": int(
            ((open_ > high + eps) | (open_ < low - eps)).sum()
        ),
        "non-positive price": int((frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()),
        "non-positive volume": int((frame["Volume"] <= 0).sum()),
        "missing price": int(frame[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()),
    }


def _adjustment_is_consistent(history: PriceHistory, tol: float = 2e-3) -> bool | None:
    """Does the total-return series differ from the price series only by dividends?

    With no dividends the two must be identical, which pins the split handling: if a
    split were applied to one series and not the other, a 3-for-1 shows up as a 3x gap.

    With dividends the check is on returns, not levels. Total return should exceed price
    return on each ex-date by roughly the dividend yield. Returns None when there is not
    enough to compare rather than claiming a pass.
    """
    close = history.close
    total = history.total_return_close
    if total is None or total.isna().all():
        return None
    # Rename before concat: both series are often called "Close", and duplicate column
    # labels make frame["Close"] return a DataFrame, which then fails on bool().
    aligned = pd.concat(
        [close.rename("price"), total.rename("total")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return None

    if history.dividends.sum() == 0.0:
        ratio = aligned["total"] / aligned["price"]
        # A pure split mismatch makes this ratio jump; a constant ratio is fine.
        return bool(ratio.max() / ratio.min() - 1.0 < tol)

    price_ret = aligned["price"].pct_change().dropna()
    total_ret = aligned["total"].pct_change().dropna()
    div_yield = (
        history.dividends.reindex(aligned.index).fillna(0.0)
        / aligned["price"].shift(1)
    ).dropna()
    common = price_ret.index.intersection(total_ret.index).intersection(div_yield.index)
    residual = (total_ret[common] - price_ret[common] - div_yield[common]).abs()
    return bool(residual.max() < tol)


def assess(
    history: PriceHistory,
    calendar: str = "XNYS",
    tail_closes: pd.Series | None = None,
    tail_source: str | None = None,
    asof: pd.Timestamp | None = None,
    requested_start=None,
    requested_end=None,
    rf: pd.Series | None = None,
    rf_had_print: pd.Series | None = None,
) -> DataQualityReport:
    """Run every check and return the report. Never raises on bad data; reports it."""
    frame = history.frame
    if len(frame) == 0:
        raise ValueError("cannot assess an empty price history")

    index = pd.DatetimeIndex(frame.index)
    first, last = index[0], index[-1]
    # The expected session set is derived from the data's own extent, which is why the
    # requested window has to be supplied separately: without it, truncation is
    # structurally invisible. A 5-year series front-truncated to 2 years reports
    # rows=520, expected=520, clean=True, because both sides shrink together.
    expected = sessions(first, last, calendar=calendar)

    report = DataQualityReport(
        ticker=history.ticker,
        source=history.source,
        fetched_at=history.fetched_at,
        first_session=first,
        last_session=last,
        rows=len(frame),
        expected_sessions=len(expected),
        missing_sessions=list(expected.difference(index)),
        unexpected_rows=list(index.difference(expected)),
        duplicate_rows=list(index[index.duplicated(keep="first")].unique()),
        bad_bars=_check_bars(frame),
        splits={
            str(day.date()): float(value)
            for day, value in history.splits[history.splits > 0].items()
        },
        dividend_total=float(history.dividends.sum()),
        adjustment_consistent=_adjustment_is_consistent(history),
    )

    # Coverage against what was actually asked for.
    if requested_start is not None:
        report.requested_start = pd.Timestamp(requested_start).normalize()
        head = sessions(report.requested_start, first, calendar=calendar)
        report.missing_head_sessions = max(len(head) - 1, 0)
    if requested_end is not None:
        report.requested_end = pd.Timestamp(requested_end).normalize()
        cutoff = min(report.requested_end, last_completed_session(asof, calendar=calendar))
        if cutoff > last:
            tail_gap = sessions(last, cutoff, calendar=calendar)
            report.missing_tail_sessions = max(len(tail_gap) - 1, 0)

    if rf is not None and len(rf) > 0:
        report.rf_sessions_total = len(rf)
        report.rf_sessions_with_print = (
            int(rf_had_print.sum()) if rf_had_print is not None else None
        )
        # A mis-scaled print (540 read as 5.40) yields a 185% rate, silently.
        report.rf_out_of_bounds = int(((rf < -0.01) | (rf > 0.25)).sum())

    log_ret = np.log(history.close / history.close.shift(1)).dropna()
    breaches = log_ret[log_ret.abs() > OUTLIER_LOG_RETURN]
    report.outliers = [(day, float(np.expm1(v))) for day, v in breaches.items()]
    report.unexplained_outliers = list(report.outliers)

    # The actual split cross-reference. This used to be a comment claiming a check that
    # did not exist: `unexplained_outliers` was simply a copy of `outliers`, so split
    # consistency was enforced only by the generic 40% threshold. Anything under
    # e^0.40 = 1.4918 escaped entirely, which means a 5-for-4 (1.25), a 4-for-3 (1.333),
    # a 7-for-5 (1.40) and every 5% stock dividend passed as CLEAN while the report
    # cheerfully printed the split it had failed to check.
    #
    # Yahoo adjusts retroactively, so a correctly-folded split leaves NO discontinuity.
    # A return on the split date shaped like the ratio is therefore evidence the feed
    # did not fold it in, at any ratio, however small.
    for day, ratio in history.splits[history.splits > 0].items():
        if ratio <= 0 or day not in log_ret.index:
            continue
        effective = ratio if ratio > 1.0 else 1.0 / ratio  # reverse splits too
        if effective < SPLIT_MIN_DETECTABLE_RATIO:
            report.notes.append(
                f"split {ratio:g} on {day.date()} is too small to verify: the expected "
                "discontinuity is inside normal daily noise"
            )
            continue
        observed = float(log_ret.loc[day])
        expected_if_unadjusted = -np.log(ratio)  # price would drop by the ratio
        if abs(observed - expected_if_unadjusted) < SPLIT_MATCH_TOLERANCE * abs(
            expected_if_unadjusted
        ):
            report.unadjusted_splits.append(
                (str(day.date()), float(ratio), float(np.expm1(observed)))
            )

    # Freshness. An in-progress bar should already have been dropped upstream; if one
    # survived to here, say so loudly.
    cutoff = last_completed_session(asof, calendar=calendar)
    if last > cutoff:
        report.bad_bars["bar for a session that has not closed"] = 1
    elif last < cutoff:
        stale_by = len(sessions(last, cutoff, calendar=calendar)) - 1
        if stale_by > 0:
            report.notes.append(
                f"last bar is {stale_by} completed session(s) behind "
                f"{cutoff:%Y-%m-%d}"
            )

    if tail_closes is not None and len(tail_closes) > 0:
        report.tail_source = tail_source or "second source"
        ours = history.close.reindex(tail_closes.index).dropna()
        common = ours.index.intersection(tail_closes.index)
        report.tail_checked = len(common)
        for day in common:
            mine, theirs = float(ours[day]), float(tail_closes[day])
            if theirs and abs(mine - theirs) / theirs > TAIL_TOLERANCE:
                report.tail_disagreements.append((day, mine, theirs))
        if report.tail_checked == 0:
            report.notes.append(
                f"{report.tail_source} returned data but no overlapping sessions"
            )
    else:
        report.notes.append(
            "no independent source corroborated the recent tail on this run"
        )

    shortfall = report.missing_head_sessions + report.missing_tail_sessions
    if shortfall > SHORTFALL_SESSIONS:
        report.notes.append(
            f"SHORT OF REQUEST by {shortfall} session(s) "
            f"({report.missing_head_sessions} at the start, "
            f"{report.missing_tail_sessions} at the end). This is not necessarily an "
            "error, a recent listing is legitimately short, but any 'N-year' label on "
            "this series is wrong."
        )
    elif requested_start is None and requested_end is None:
        report.notes.append(
            "requested window not supplied, so shortness could not be checked"
        )

    if history.dividends.sum() == 0.0:
        report.notes.append(
            "no dividends in window, so price and total-return series coincide"
        )
    report.notes.append(
        "prices are retroactively split-adjusted; true as-traded levels are "
        "unavailable from free sources. Strikes are struck in the same adjusted "
        "space, matching how the OCC adjusts open contracts."
    )
    report.notes.append(
        "full-window independence is NOT verified: no free no-key second source "
        "covers five years. Only the recent tail is externally corroborated."
    )
    return report
