---
name: verify
description: Use before reporting any change in company-analyst as done, or before committing. Runs the pytest suite and a CLI import smoke check; read the known-failure note before assuming a failure is yours.
---

# Verify company-analyst

Run from `~/code/company-analyst`. Use the repo venv (`make venv` builds it); never system python.

1. `make test` (same as `.venv/bin/pytest -q`). About 5 s. Pass = 0 failed. Last observed 2026-11-16: `419 passed`.
2. `.venv/bin/canalyst --help`. Pass = usage listing `backtest` and `sweep`. Proves the entry point imports.
3. Changes to backtest, option pricing, vol, or strategy logic: run the matching file alone first (`.venv/bin/pytest -q tests/test_backtest.py` etc.), since the README says every sharp edge has a pinning test.

## Real run (only when asked)
`make run` / `canalyst backtest --ticker TSLA` call yfinance (network) and write PNGs into `out/`. Do not run as a default smoke check.

## Gotchas
- Tests that check cache freshness must stamp `fetched_at` with the current time. A hardcoded date ages past the 12 h TTL the next day and the cache test fails for no code reason (fixed 2026-11-16 in `test_loader_fetches_then_serves_from_cache`).
- yfinance deprecation warnings are filtered in `pyproject.toml`; do not widen the filter to hide new ones.
- Tests use stub providers and `tests/fixtures/`; a test that needs network is a bug.
- There is no linter or type checker configured.

## Rules
If a check still fails after three fix attempts, stop and report to the user instead of retrying. Never skip a test, retry until green, or edit a test to make it pass.
