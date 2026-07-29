"""Independent close prices for the recent tail, from api.nasdaq.com.

Scope, measured rather than assumed: this endpoint serves roughly the last ten
sessions. A request for 2024-08 returns `totalRecords: 0`. So it cannot corroborate a
five-year history, and this module does not pretend to be a `PriceHistoryProvider`.

What it is genuinely good for is the question "is the newest data right", which is
where staleness and partial-bar errors live. It independently answers:
  - which session was the last one to actually close
  - what that session's close really was

Stooq used to fill the full-history role. As of 2026-07-29 it answers with a JavaScript
proof-of-work bot challenge instead of CSV, so it is not usable and no attempt is made
to work around it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
HISTORICAL_URL = (
    "https://api.nasdaq.com/api/quote/{ticker}/historical"
    "?assetclass=stocks&fromdate={frm}&todate={to}&limit={limit}"
)


def _parse_money(raw: str | None) -> float | None:
    """Nasdaq returns prices as '$307.44'. Anything unparseable becomes None."""
    if not raw:
        return None
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


class NasdaqTailProvider:
    name = "nasdaq"
    #: Beyond roughly this far back the endpoint returns nothing.
    max_lookback_days = 14

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._session = requests.Session()

    def recent_closes(
        self, ticker: str, lookback_days: int | None = None
    ) -> pd.Series | None:
        """Daily closes for the recent tail, or None if the endpoint is unreachable.

        Returns a date-indexed Series. Never raises: a second source being down is a
        reason to downgrade a check to a warning, not to abort a run.
        """
        span = lookback_days or self.max_lookback_days
        today = dt.date.today()
        url = HISTORICAL_URL.format(
            ticker=ticker.upper(),
            frm=today - dt.timedelta(days=span),
            to=today,
            limit=span * 2,
        )
        try:
            response = self._session.get(
                url,
                headers={"User-Agent": BROWSER_UA, "Accept": "application/json"},
                timeout=self._timeout,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        rows = (
            ((payload.get("data") or {}).get("tradesTable") or {}).get("rows") or []
        )
        closes: dict[pd.Timestamp, float] = {}
        for row in rows:
            price = _parse_money(row.get("close"))
            raw_date = row.get("date")
            if price is None or not raw_date:
                continue
            try:
                stamp = pd.Timestamp(dt.datetime.strptime(raw_date, "%m/%d/%Y"))
            except ValueError:
                continue
            closes[stamp.normalize()] = price

        if not closes:
            return None
        series = pd.Series(closes, name="nasdaq_close").sort_index()
        series.index = pd.DatetimeIndex(series.index)
        return series
