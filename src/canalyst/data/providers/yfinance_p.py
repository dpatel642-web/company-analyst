"""Primary price provider: Yahoo via yfinance.

yfinance is a scraper over an undocumented endpoint. Its dangerous failure mode is not
an exception, it is a plausible-looking series with a wrong adjustment factor. Nothing
in this module can detect that; `quality.py` is what tries.

Two shape quirks handled here, both verified against yfinance 1.5.2:
  - columns come back as a MultiIndex even for a single ticker
  - the index is tz-aware, so it will not compare equal to a naive session index
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..prices import PriceHistory


class YFinanceProvider:
    name = "yfinance"

    def history(self, ticker: str, start, end) -> PriceHistory | None:
        # yfinance treats `end` as exclusive; pad a day so `end` itself is included.
        end_exclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        fetched_at = pd.Timestamp.now("UTC").tz_localize(None)

        try:
            raw = yf.download(
                ticker,
                start=pd.Timestamp(start).normalize(),
                end=end_exclusive,
                auto_adjust=False,  # keep dividends out of the price series
                actions=True,  # need the Dividends / Stock Splits columns
                progress=False,
                threads=False,
            )
        except Exception:
            return None

        if raw is None or len(raw) == 0:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(-1)

        required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not required.issubset(raw.columns):
            return None

        idx = pd.DatetimeIndex(raw.index)
        raw.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()

        frame = pd.DataFrame(
            {
                "Open": raw["Open"].astype(float),
                "High": raw["High"].astype(float),
                "Low": raw["Low"].astype(float),
                "Close": raw["Close"].astype(float),
                "Volume": raw["Volume"].astype(float),
                "Dividends": raw.get(
                    "Dividends", pd.Series(0.0, index=raw.index)
                ).fillna(0.0).astype(float),
                "Splits": raw.get(
                    "Stock Splits", pd.Series(0.0, index=raw.index)
                ).fillna(0.0).astype(float),
            },
            index=raw.index,
        )
        total_return_close = raw["Adj Close"].astype(float)
        total_return_close.name = "TotalReturnClose"

        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        total_return_close = total_return_close.reindex(frame.index)
        if len(frame) == 0:
            return None

        return PriceHistory(
            ticker=ticker.upper(),
            frame=frame,
            total_return_close=total_return_close,
            source=self.name,
            fetched_at=fetched_at,
        )
