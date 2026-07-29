"""Daily price history behind a swappable provider, with a self-dating cache.

What "adjusted" means here, because getting this wrong silently poisons everything
downstream:

Yahoo returns prices that are *retroactively split-adjusted* whatever you ask for.
True as-traded prices are not available from any free source. TSLA closed near $891
on 2022-08-24 and near $296 the next day after a 3-for-1 split, yet the series shows
no discontinuity at all.

That turns out to be the right basis rather than a defect. When a stock splits 3-for-1
the OCC adjusts every open option contract the same way, dividing the strike by three
and tripling the multiplier. A backtest run entirely in split-adjusted space therefore
matches what would really have happened to the contract, provided it is *consistent*.
Consistency is what `quality.py` verifies.

Two series come back, and they are not interchangeable:
  frame["Close"]      split-adjusted, dividend-UNadjusted. Strikes and assignment.
  total_return_close  split- and dividend-adjusted. The buy-and-hold benchmark.

Using the total-return series to strike options would quietly move every strike by the
cumulative dividend yield.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .calendar import last_completed_session

OHLCV = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Splits"]
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache"
DEFAULT_TTL_HOURS = 12.0


@dataclass(frozen=True)
class PriceHistory:
    """One ticker's daily history, tagged with where and when it came from."""

    ticker: str
    frame: pd.DataFrame
    total_return_close: pd.Series
    source: str
    fetched_at: pd.Timestamp

    @property
    def close(self) -> pd.Series:
        """Split-adjusted, dividend-unadjusted close. The option-mechanics series."""
        return self.frame["Close"]

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)

    @property
    def dividends(self) -> pd.Series:
        return self.frame["Dividends"]

    @property
    def splits(self) -> pd.Series:
        return self.frame["Splits"]

    def age_hours(self, now: pd.Timestamp | None = None) -> float:
        # Timestamp.utcnow() is deprecated in pandas 3 and removed in 4.
        ref = (
            pd.Timestamp(now)
            if now is not None
            else pd.Timestamp.now("UTC").tz_localize(None)
        )
        if ref.tz is not None:
            ref = ref.tz_convert(None)
        return (ref - self.fetched_at).total_seconds() / 3600.0

    def slice(self, start, end) -> "PriceHistory":
        lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        mask = (self.frame.index >= lo) & (self.frame.index <= hi)
        return PriceHistory(
            ticker=self.ticker,
            frame=self.frame.loc[mask].copy(),
            total_return_close=self.total_return_close.loc[mask].copy(),
            source=self.source,
            fetched_at=self.fetched_at,
        )


class PriceHistoryProvider(Protocol):
    """Mirrors the provider seam in edgar-screener: every failure returns None.

    A provider that raises forces every caller into try/except. A provider that
    returns None lets the caller decide whether this source was load-bearing.
    """

    name: str

    def history(self, ticker: str, start, end) -> PriceHistory | None: ...


# ------------------------------------------------------------------------- caching


def _cache_paths(cache_dir: Path, ticker: str, source: str) -> tuple[Path, Path]:
    stem = f"{ticker.upper()}__{source}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"


def _write_cache(cache_dir: Path, history: PriceHistory) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(cache_dir, history.ticker, history.source)
    out = history.frame.copy()
    out["TotalReturnClose"] = history.total_return_close
    out.to_csv(csv_path, index_label="Date")
    meta_path.write_text(
        json.dumps(
            {
                "ticker": history.ticker,
                "source": history.source,
                "fetched_at": history.fetched_at.isoformat(),
                "rows": int(len(out)),
                "first": str(history.frame.index[0].date()),
                "last": str(history.frame.index[-1].date()),
            },
            indent=2,
        )
    )


def _read_cache(cache_dir: Path, ticker: str, source: str) -> PriceHistory | None:
    csv_path, meta_path = _cache_paths(cache_dir, ticker, source)
    if not (csv_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        frame = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    except (OSError, ValueError, KeyError):
        return None
    if "TotalReturnClose" not in frame.columns:
        return None
    total_return = frame.pop("TotalReturnClose")
    # Strip the index name the CSV round-trip introduces, so a cache hit is
    # indistinguishable from a fresh provider fetch rather than merely similar.
    frame.index = pd.DatetimeIndex(frame.index).normalize().rename(None)
    total_return.index = frame.index
    return PriceHistory(
        ticker=meta["ticker"],
        frame=frame,
        total_return_close=total_return,
        source=meta["source"],
        fetched_at=pd.Timestamp(meta["fetched_at"]),
    )


# -------------------------------------------------------------------------- loading


def drop_incomplete_bar(
    history: PriceHistory, asof: pd.Timestamp | None = None
) -> PriceHistory:
    """Remove any bar for a session that has not closed yet.

    Verified live on 2026-07-29: yfinance returned a row for that date while the most
    recent completed session was 2026-07-28. That row was an in-progress bar. Left in,
    it becomes the final "close" of the series and every terminal number inherits it.
    """
    if len(history.frame) == 0:
        return history
    cutoff = last_completed_session(asof)
    if history.frame.index[-1] <= cutoff:
        return history
    keep = history.frame.index <= cutoff
    return PriceHistory(
        ticker=history.ticker,
        frame=history.frame.loc[keep].copy(),
        total_return_close=history.total_return_close.loc[keep].copy(),
        source=history.source,
        fetched_at=history.fetched_at,
    )


def load_history(
    ticker: str,
    start,
    end,
    provider: PriceHistoryProvider | None = None,
    cache_dir: Path | None = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    drop_incomplete: bool = True,
    use_cache: bool = True,
) -> PriceHistory:
    """Fetch daily history, preferring a fresh cache entry.

    Raises rather than returning a short series when the provider fails: a silently
    truncated price history is worse than a stack trace.
    """
    if provider is None:
        from .providers.yfinance_p import YFinanceProvider

        provider = YFinanceProvider()
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE

    if use_cache:
        cached = _read_cache(cache_dir, ticker, provider.name)
        if cached is not None and cached.age_hours() <= ttl_hours:
            sliced = cached.slice(start, end)
            if len(sliced.frame) > 0:
                return drop_incomplete_bar(sliced) if drop_incomplete else sliced

    fetched = provider.history(ticker, start, end)
    if fetched is None or len(fetched.frame) == 0:
        raise RuntimeError(
            f"provider {provider.name!r} returned no history for {ticker!r} "
            f"over {start}..{end}"
        )
    if use_cache:
        _write_cache(cache_dir, fetched)
    return drop_incomplete_bar(fetched) if drop_incomplete else fetched
