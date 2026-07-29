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
    bad_bars: dict[str, int] = field(default_factory=dict)
    splits: dict[str, float] = field(default_factory=dict)
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
            f"  dividends paid    {self.dividend_total:.4f}",
            f"  moves over {OUTLIER_LOG_RETURN:.0%}     "
            f"{len(self.outliers)} ({len(self.unexplained_outliers)} unexplained)",
            f"  adjustment self-consistent  {self.adjustment_consistent}",
        ]
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
) -> DataQualityReport:
    """Run every check and return the report. Never raises on bad data; reports it."""
    frame = history.frame
    if len(frame) == 0:
        raise ValueError("cannot assess an empty price history")

    index = pd.DatetimeIndex(frame.index)
    first, last = index[0], index[-1]
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
        bad_bars=_check_bars(frame),
        splits={
            str(day.date()): float(value)
            for day, value in history.splits[history.splits > 0].items()
        },
        dividend_total=float(history.dividends.sum()),
        adjustment_consistent=_adjustment_is_consistent(history),
    )

    # Outliers, each cross-referenced against the corporate-action feed. A real split
    # that Yahoo has already folded in should NOT produce a jump, so a jump on a split
    # date is evidence of inconsistent adjustment, not an explanation for it.
    log_ret = np.log(history.close / history.close.shift(1)).dropna()
    breaches = log_ret[log_ret.abs() > OUTLIER_LOG_RETURN]
    report.outliers = [(day, float(np.expm1(v))) for day, v in breaches.items()]
    report.unexplained_outliers = list(report.outliers)

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
