# Backlog

Open defects from an adversarial review run 2026-07-29 (three agents: options/accounting,
data layer, statistics). Six were fixed in `a563d49`; these are the survivors, ranked.
Each has a concrete failure scenario, because a backlog item without one is a guess.

**Do these before running a multi-ticker batch.** Every one of the top four biases either
dividend payers or recent listings, and the intended universe is ~35 watchlist names that
are mostly both.

---

## 1. Options price with `q = 0` while dividends are credited separately

`backtest.py` (`_mark` calls, both sites), `covered_call.py` / `protective_put.py`
(`strike_from_delta` calls).

`bs.py` supports a continuous dividend yield and the module docstring lists it as a
project convention, but the engine never passes it. The model's forward is `S`, not
`S·e^{-qT}`, on a series that is deliberately dividend-*un*adjusted and on which dividend
cash is separately credited. So the writer collects a premium priced as if no dividend
were coming **and** receives the dividend. The bias is not refunded at expiry, because
intrinsic value is `q`-independent.

Measured at PG-like parameters (S=160, T=21/252, r=4.2%, sigma=18%, q=2.4%): premium
booked 1.2091 against a fair 1.1311, so **+4.9bp of spot per cycle, about 59bp/year**, and
141bp/year at a 6% yield. The chosen strike's true delta is 0.2375, not the pre-specified
0.25. The protective put is flattered in the same direction, which defeats the stated
purpose of the put arm as a sanity check.

**Fix:** derive a trailing-12-month dividend yield series, thread it through
`BarContext`, `_mark`, and both `strike_from_delta` call sites.
**Trap:** `test_expiring_call_settles_at_intrinsic` recomputes the expected premium with
`q` defaulted to 0, so it actively pins the wrong convention. Fixing the engine breaks
that test, and the test is what needs changing.

## 2. A short provider response is laundered into a wide cache entry

`prices.py` `_read_cache` coverage check.

The check compares `requested_start`/`requested_end` against the new request and never
consults the frame it is vouching for, even though the metadata now records the actual
extent. A provider returning two years for a five-year request gets stamped with the
five-year range and thereafter satisfies five-year requests.

Live, no corruption needed: `RDDT` requested 2021-03-01..2026-07-29, returned
2024-03-21..2026-07-28, 589 rows, `assess` reported clean. `ARM` likewise at 719 rows.
`sweep` then prints a table headed "5y window" built from 2.4 and 2.9 years with no
per-ticker window column.

**Watchlist exposure:** SNDK, BMNR, IREN, RVI, SPCX, DRAM are all recent enough to hit
this. A legitimately short history is fine; silently reporting it as five years is not.
**Fix:** compare `data_first`/`data_last` against the request, and pass the requested
window into `assess` so shortness is a reported field rather than invisible. Also make
the two-file cache write atomic (temp plus rename); interrupting between the CSV and the
JSON currently leaves a narrow frame under a wide stamp.

## 3. `assess` derives the expected session set from the data's own extent

`quality.py` `assess`.

`expected = sessions(first, last)` where both come from the frame, so truncation is
structurally invisible: a 5y series front-truncated to 2y reports `rows=520
expected=520 clean=True`. `rows` and `expected_sessions` are rendered but never compared
in `failures`, so a duplicated session row also passes (`missing` and `unexpected` are set
differences, both empty for a duplicate). `cmd_sweep` discards the report entirely and
keeps only the boolean.

## 4. A truncated `^IRX` response becomes a five-year constant

`riskfree.py` `risk_free_series`.

`strict=True` guards a total fetch failure but not a short one. `fetch_irx` accepts any
series of length >= 1, then ffill/bfill extend it across the whole index with no bound and
no note. A response covering only 2021 pins the rate at 0.05% for five years against a
true mean of 3.5436% — a 3.5pp one-signed error that overstates Sharpe by ~0.35 at 10%
vol and **alone flips the `sharpe_over_1` verdict**. No bounds check either: a mis-scaled
print of `540` for `5.40` yields 185.6% silently. `DataQualityReport` has no rf fields at
all, making this the only series in the layer with zero integrity checking.

Verified not currently realised: 1254 of 1254 sessions have a real print, min 0.0030%, max
5.3480%, no gaps over 5 days. The exposure is the unbounded short response.

## 5. Splits are collected but never cross-referenced

`quality.py` — `report.unexplained_outliers = list(report.outliers)`.

The comment claims outliers are "each cross-referenced against the corporate-action feed".
No cross-reference exists. Split consistency is enforced only by the generic 40%
log-return threshold, so anything below `e^0.40 = 1.4918` escapes: a 5-for-4 (1.25), a
4-for-3 (1.333), a 7-for-5 (1.40) and any 5% stock dividend all pass with zero outliers,
`adjustment_consistent=True`, verdict CLEAN, while the report cheerfully prints the split
it failed to check. Every pre-split strike is then wrong by the split ratio, which
inverts assignment across that whole segment.

`test_unapplied_split_is_caught` passes only because 3 > 1.4918, so it is exercising the
threshold, not split consistency. The report already holds both required facts and never
multiplies them.
**Fix:** for each `(day, ratio)` in `history.splits`, assert
`|log(close[day]/close[day-1])|` is NOT near `|log(1/ratio)|`.

## 6. The invariant is telescoping, so interim mark errors are invisible to it

`backtest.py` marking loop.

`sum(mark_t - mark_{t-1})` collapses to `mark_settle - mark_open`, so everything between
cancels. A wrong interim time-to-expiry changes the daily return path, volatility, Sharpe
and drawdown while leaving terminal value **bit-identical** and every residual at 1e-13.

Four mutants touching only the interim mark were run against the 27 load-bearing
assertions. An interim mark carrying **zero theta** failed 0 of 27 while moving Sharpe by
0.024 and vol by 1.3pp. Since `sweep` reports `sharpe_over_1` as a hard boolean and a
measured BH Sharpe sat at 0.979, an error this size flips that bit.

**Fix:** mark against `crosscheck.crr_price` on a sample of bars. This also catches any
pricing-argument error, including item 1.

## 7. Smaller, all with concrete repros

- **`fee_per_contract` is charged per spec, not per contract.** At `shares=100,
  contracts=100`, 35 rolls charges 22.75 instead of 2275.00, a 100x understatement.
  Hidden at unit size, appears the moment notional scales.
- **`ProtectivePut` borrows to buy the put.** `target_shares` sweeps cash to zero, then
  `options_to_open` buys a put with money that no longer exists: min cash -2.11, negative
  on 85.3% of bars. The `NEG_CASH_TOL` guard is applied to `CoveredCall` only.
- **`assert_identity` is blind to NaN.** `worst.max() > tol` uses pandas `skipna=True`, so
  all-NaN yields `nan > tol == False`. One NaN in `rate` at bar 400 of 756 truncates
  `summarise` by 17 months (via its `dropna`) and reports cum -36.25% / Sharpe -0.76
  against a true -46.41% / -0.52, with `assert_identity` **and** `verify` both passing.
- **`starting_equity <= 0` is accepted** and the collateralisation guard fails open:
  `starting_equity=0.0` holds 1.0 share against cash of -101.02 and writes a call against
  it, identity passing. Bar 0's residual is hardcoded to zero, which exempts
  initialisation from the only check.
- **`rolls` counts cycles where nothing was written.** At `sigma = 0` everywhere:
  `rolls=35`, `net_premium=0`, `max(open_positions)=0`.
- **Non-unique index settles at time value, not intrinsic.** `position_of_index` keeps the
  last occurrence, so a duplicated expiry row banks a time-value mark. One duplicated bar
  moved terminal value by +0.055% with identity passing. Latent: the yfinance provider
  dedupes, but the checker is provider-agnostic by design.
- **The adjustment check's power is proportional to dividend yield.** `tol = 2e-3` on a
  per-ex-date return means a detection floor near 0.8% annualised, so a dropped or
  day-shifted dividend passes as consistent for AAPL, GOOGL, META, MA, NVDA. Also
  `total_return_close` has no completeness check: 122 NaNs of 124 still reports consistent.
- **The `^IRX` conversion error is understated about 2x and is one-signed.** `riskfree.py`
  claims ~8bp at 4%; the true continuously-compounded bond-equivalent yield differs by
  15.4bp at 4% and 25.3bp at the window's 5.4% peak, low on 1254 of 1254 days. The choice
  is defensible, the stated magnitude is not.
- **Hard failures with no override push users to disable the gate.** Non-positive volume
  is a hard failure, so `assess` can never pass an index ticker (Yahoo reports volume 0
  for `^GSPC`, `^VIX`), and every real >40% move is by construction "unexplainable". Wants
  a warning tier or an acknowledged-exceptions list.
- **The second source's freshness answer is fetched and discarded.** `common = ours.index
  .intersection(tail_closes.index)` drops sessions Nasdaq has that we lack, so "Nasdaq
  knows about a session we are missing" is not a disagreement. The staleness check derives
  its cutoff from the same calendar the primary path uses, so there is no genuinely
  independent freshness check anywhere.
- **`cli.py` never corroborates.** Both `backtest` and `sweep` call `assess` with no
  `tail_closes`, so the integrity gate runs with zero external corroboration and still
  returns clean. Only `scripts/fin642_run.py` wires the tail in.

---

## WRITEUP.md corrections gathered but not yet applied

Numbers verified 2026-07-29; the file still contains the originals.

| claim | in the file | correct |
|---|---|---|
| IV x1.20 covered call | "+63.3%, Sharpe 0.36" | **+34.59%, Sharpe 0.28** at the pre-specified 0.25 delta, which is **still below** buy-and-hold's +36.17%. The quoted pair took the 0.25-delta loss as baseline and the 0.40-delta cell as the gain, and +63.28% is the argmax of all nine cells on both metrics |
| Sharpe 1.0 requirement | "roughly 47% annualised excess" | **44% excess** (47.5% total). The word "excess" is load-bearing in a sentence about exactly that distinction |
| T-bill range | "0.02% to 5.21%" | those are the `ln(1+r)` internals. The bill's own quotes are mean **3.62%**, min 0.02%, max **5.35%** |
| leverage pairing | "-240 against a share worth 222" | min cash -240.49 occurred on **2026-01-15 with spot at 438.57** (exposure 2.21x). Max exposure over the run is 4.37x, on a different day. Also stated in `covered_call.py` and a test docstring. The 93.72% vs 59.67% vol half reproduces exactly |
| "ten large caps" | asserted | no such list exists in the repo; `--tickers` has no default. Record the universe or weaken the claim |
| Sortino | 0.52 / 0.02 / 0.68 | recompute under target downside deviation: **0.50 / 0.02 / 0.67** |
| protective put edge | presented as a general property | it is **one year**. Beat buy-and-hold in **2 of 6** calendar years; 2022 alone contributed +22.05pp. Excluding 2022: buy-and-hold **+289.40%** vs put **+227.98%** |
| "nothing does / no overlay closes it" | asserted | a universal over 10 tested configurations. Directionally very likely, stated as proven |
| "Every number is produced by a tested library" | asserted | the 112.1%, the sensitivity grids, and the prose arithmetic (51pp, 23pp, "roughly 47%") are inline. Defect 5 above came from exactly that |

## Then: the batch build

Agreed scope: union of all three watchlists (~35 equities), screened for optionability
with exclusions reported; VRP measured from real expired-contract option prices on a
subset and used to calibrate the rest; and the strategies still to add are cash-secured
put, bull call spread, long straddle, collar plus zero-cost collar, short strangle, iron
condor, and the wheel.

### Real historical implied vol: PARTIALLY RETRACTED, tested 2026-07-29

An earlier note in this file claimed this source was "confirmed working" and would remove
the project's central limitation. **That was based on a single TSLA sample and does not
generalise.** Retracted, with the measurements.

What is true: expired contracts are queryable (`get_option_instruments(state='expired')`)
and `get_option_historicals` returns daily OHLC for them. `TSLA 240816C00250000` closed at
12.18 on 2024-07-19, a real roll date, and inverts cleanly.

What is false: that the data is generally usable. Sampled five WMT roll dates spanning
2022-2025 at the 0.25-delta strike, inverted with `bs.implied_vol`:

| roll | K(adj) | traded close | RV | implied vol | IV/RV | quality |
|---|---|---|---|---|---|---|
| 2022-05-20 | 43.33 | 0.217 | 35.0% | 28.6% | 0.82 | low 0.01, range 0.01-2.88 |
| 2024-02-16 | 58.33 | 2.320 | 13.1% | 47.1% | **3.60** | both bars `interpolated=true` |
| 2024-09-20 | 82.00 | 0.010 | 19.3% | 5.9% | **0.30** | 0.01 = min tick |
| 2025-04-17 | 101.00 | 0.010 | 37.9% | 11.7% | **0.31** | 0.01 = min tick |
| 2025-11-21 | 110.00 | 0.850 | 22.1% | 20.9% | 0.95 | real, range 0.78-2.03 |

Ratios span 0.30 to 3.60, a factor of twelve. Three of five bars are `interpolated=true`
(the API's own gap-fill flag, which its own guide says to ignore for analytics) and/or
pinned at the $0.01 minimum tick, which means no trades occurred rather than a price of
one cent. A 3.7%-OTM WMT call with 20 days at 19% realised vol is worth roughly $0.80.
Only two observations are plausible, both **below** 1.0, which would imply IV under
realised vol, the opposite of a variance risk premium. At n=2 with this noise that is not
a finding either.

TSLA sampled cleanly because its options are exceptionally liquid. Thin OTM contracts on
ordinary large caps do not, and coverage degrades going back.

**Where this leaves the VRP.** `bs.implied_vol` is built and tested (bisection, declines
when vega has collapsed rather than guessing), so the *inverter* is not the problem, the
*source* is. Options worth trying, in order: (a) filter hard on data quality, taking only
non-interpolated bars above a few times the minimum tick, and accept far fewer
observations with an honest confidence interval; (b) invert near-the-money contracts
instead, which are liquid enough to price reliably, and accept that the measured VRP is
then an ATM number applied to an OTM strike; (c) use a keyed vendor with real historical
IV. Until one of those lands, the realised-vol basis stands and its understatement stays
disclosed rather than silently corrected.

**Two data traps found while testing this, both worth keeping:**
- **A split creates a second option chain.** WMT split 3-for-1 on 2024-02-26. Contracts
  expiring *before* it live under `chain_id fa2f0d5e...` with **as-traded** strikes (130.00),
  while everything after lives under `69632d04...` with post-split strikes. Pre-split
  prices must be divided by the split ratio to compare against a split-adjusted spot.
- **`get_option_chains` only returns the active chain**, so the pre-split chain is
  discoverable only through the expired-instruments endpoint. Nothing warns you.

Budget if retried: roughly 60-120 calls per ticker across five years, cached aggressively,
and expect to discard a large fraction of the bars.
