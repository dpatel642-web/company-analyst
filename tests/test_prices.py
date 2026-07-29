"""Price container, cache, and loader. A stub provider keeps this offline."""

from __future__ import annotations

import pandas as pd
import pytest

from canalyst.data.prices import (
    PriceHistory,
    _read_cache,
    _write_cache,
    load_history,
)

ASOF = pd.Timestamp("2026-07-29T14:00:00", tz="UTC")


class StubProvider:
    """Counts calls so tests can prove the cache was or was not consulted."""

    name = "stub"

    def __init__(self, history: PriceHistory | None) -> None:
        self._history = history
        self.calls = 0

    def history(self, ticker, start, end):
        self.calls += 1
        return self._history


def _synthetic(last: str = "2026-07-28", n: int = 6) -> PriceHistory:
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    frame = pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 1e7,
            "Dividends": 0.0,
            "Splits": 0.0,
        },
        index=idx,
    )
    return PriceHistory(
        ticker="TEST",
        frame=frame,
        total_return_close=close.rename("TotalReturnClose"),
        source="stub",
        fetched_at=pd.Timestamp("2026-07-29T12:00:00"),
    )


# ----------------------------------------------------------------------- the container


def test_accessors_expose_the_right_series():
    h = _synthetic()
    assert h.close.equals(h.frame["Close"])
    assert list(h.sessions) == list(h.frame.index)
    assert h.dividends.sum() == 0.0
    assert h.splits.sum() == 0.0


def test_slice_narrows_both_series_together():
    h = _synthetic(n=10)
    lo, hi = h.frame.index[2], h.frame.index[5]
    s = h.slice(lo, hi)
    assert len(s.frame) == 4
    assert len(s.total_return_close) == 4
    assert s.frame.index[0] == lo and s.frame.index[-1] == hi
    assert s.source == h.source and s.fetched_at == h.fetched_at


def test_age_hours_measures_from_fetch_time():
    h = _synthetic()
    assert h.age_hours(pd.Timestamp("2026-07-29T18:00:00")) == pytest.approx(6.0)


# ------------------------------------------------------------------------------- cache


def test_cache_round_trips_exactly(tmp_path):
    original = _synthetic()
    _write_cache(tmp_path, original, "2026-07-01", "2026-07-28")
    restored = _read_cache(tmp_path, "TEST", "stub")

    assert restored is not None
    pd.testing.assert_frame_equal(
        restored.frame, original.frame, check_dtype=False, check_freq=False
    )
    pd.testing.assert_series_equal(
        restored.total_return_close,
        original.total_return_close,
        check_names=False,
        check_freq=False,
    )
    assert restored.fetched_at == original.fetched_at
    assert restored.source == "stub"


def test_cache_miss_returns_none(tmp_path):
    assert _read_cache(tmp_path, "NOPE", "stub") is None


def test_corrupt_cache_returns_none_rather_than_raising(tmp_path):
    _write_cache(tmp_path, _synthetic(), "2026-07-01", "2026-07-28")
    (tmp_path / "TEST__stub.csv").write_text("not,a,valid\nframe")
    assert _read_cache(tmp_path, "TEST", "stub") is None


def test_cache_without_total_return_column_is_rejected(tmp_path):
    """An older cache layout must be treated as a miss, not silently half-loaded."""
    h = _synthetic()
    _write_cache(tmp_path, h, "2026-07-01", "2026-07-28")
    frame = pd.read_csv(tmp_path / "TEST__stub.csv", index_col="Date")
    frame.drop(columns=["TotalReturnClose"]).to_csv(
        tmp_path / "TEST__stub.csv", index_label="Date"
    )
    assert _read_cache(tmp_path, "TEST", "stub") is None


# ------------------------------------------------------------------------------ loader


def test_loader_fetches_then_serves_from_cache(tmp_path):
    provider = StubProvider(_synthetic())
    args = ("TEST", "2026-07-01", "2026-07-28")
    kwargs = dict(provider=provider, cache_dir=tmp_path, drop_incomplete=False)

    first = load_history(*args, **kwargs)
    assert provider.calls == 1 and len(first.frame) > 0

    second = load_history(*args, **kwargs)
    assert provider.calls == 1, "second call should have hit the cache"
    pd.testing.assert_frame_equal(first.frame, second.frame, check_freq=False)


def test_expired_cache_triggers_a_refetch(tmp_path):
    provider = StubProvider(_synthetic())
    args = ("TEST", "2026-07-01", "2026-07-28")
    load_history(*args, provider=provider, cache_dir=tmp_path, drop_incomplete=False)
    load_history(
        *args,
        provider=provider,
        cache_dir=tmp_path,
        ttl_hours=0.0,
        drop_incomplete=False,
    )
    assert provider.calls == 2


def test_use_cache_false_always_refetches(tmp_path):
    provider = StubProvider(_synthetic())
    for _ in range(3):
        load_history(
            "TEST", "2026-07-01", "2026-07-28",
            provider=provider, cache_dir=tmp_path,
            use_cache=False, drop_incomplete=False,
        )
    assert provider.calls == 3


def test_provider_failure_raises_rather_than_returning_a_short_series(tmp_path):
    """A truncated price history is more dangerous than a stack trace."""
    with pytest.raises(RuntimeError, match="no history"):
        load_history(
            "TEST", "2026-07-01", "2026-07-28",
            provider=StubProvider(None), cache_dir=tmp_path,
        )


def test_empty_provider_result_also_raises(tmp_path):
    empty = _synthetic()
    blank = PriceHistory(
        empty.ticker, empty.frame.iloc[0:0], empty.total_return_close.iloc[0:0],
        empty.source, empty.fetched_at,
    )
    with pytest.raises(RuntimeError, match="no history"):
        load_history(
            "TEST", "2026-07-01", "2026-07-28",
            provider=StubProvider(blank), cache_dir=tmp_path,
        )


def test_loader_drops_an_unclosed_final_bar(tmp_path, monkeypatch):
    """End-to-end version of the partial-bar guard, through the public entry point."""
    import canalyst.data.prices as prices_mod

    monkeypatch.setattr(
        prices_mod, "last_completed_session",
        lambda asof=None, **kw: pd.Timestamp("2026-07-28"),
    )
    with_today = _synthetic(last="2026-07-29", n=5)
    assert with_today.frame.index[-1] == pd.Timestamp("2026-07-29")

    loaded = load_history(
        "TEST", "2026-07-01", "2026-07-29",
        provider=StubProvider(with_today), cache_dir=tmp_path, drop_incomplete=True,
    )
    assert loaded.frame.index[-1] == pd.Timestamp("2026-07-28")


# ------------------------------------------------- coverage: the bug that actually bit


def test_cache_narrower_than_the_request_is_a_miss(tmp_path):
    """A cache entry covering three months must NOT satisfy a five-year request.

    This is the regression for a real failure. A PG entry written over H1 2024 as a test
    fixture silently answered a five-year query with 64 bars, and the resulting Sharpe
    looked perfectly reasonable. Silently short data is the failure mode this whole
    project is built to refuse.
    """
    narrow = _synthetic(last="2024-06-28", n=124)
    _write_cache(tmp_path, narrow, "2024-01-01", "2024-06-30")

    # The same narrow window is still a legitimate hit.
    assert _read_cache(tmp_path, "TEST", "stub",
                       start="2024-02-01", end="2024-06-01") is not None
    # A wider window is not.
    assert _read_cache(tmp_path, "TEST", "stub",
                       start="2021-07-29", end="2026-07-29") is None
    # Overlapping but extending past either end is also not.
    assert _read_cache(tmp_path, "TEST", "stub",
                       start="2023-01-01", end="2024-06-30") is None
    assert _read_cache(tmp_path, "TEST", "stub",
                       start="2024-01-01", end="2025-01-01") is None


def test_loader_refetches_when_the_cache_is_too_narrow(tmp_path):
    provider = StubProvider(_synthetic(last="2026-07-28", n=400))
    load_history("TEST", "2026-01-01", "2026-07-28",
                 provider=provider, cache_dir=tmp_path, drop_incomplete=False)
    assert provider.calls == 1
    # Widening the request must go back to the provider, not reuse the narrow entry.
    load_history("TEST", "2020-01-01", "2026-07-28",
                 provider=provider, cache_dir=tmp_path, drop_incomplete=False)
    assert provider.calls == 2


def test_legacy_cache_without_coverage_metadata_is_a_miss(tmp_path):
    """Entries written before coverage was tracked cannot be trusted to cover anything."""
    import json
    _write_cache(tmp_path, _synthetic(), "2026-07-01", "2026-07-28")
    meta_path = tmp_path / "TEST__stub.json"
    meta = json.loads(meta_path.read_text())
    del meta["requested_start"], meta["requested_end"]
    meta_path.write_text(json.dumps(meta))
    assert _read_cache(tmp_path, "TEST", "stub",
                       start="2026-07-01", end="2026-07-28") is None
    # With no range asked for, it still loads: callers that do not care are not punished.
    assert _read_cache(tmp_path, "TEST", "stub") is not None
