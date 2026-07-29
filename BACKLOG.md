# Backlog

Open defects from an adversarial review run 2026-07-29 (three agents: options/accounting,
data layer, statistics). Each has a concrete failure scenario, because a backlog item
without one is a guess.

## Closed

- Round one, six defects, `a563d49`: dividend-policy mismatch between benchmark and
  overlay, partial bars persisted to cache, Sortino denominator, the telescoping
  `verify()` tautology, the holiday-shifted-expiry right edge, unguarded CAGR.
- **#1 dividend yield `q`** in pricing and strike selection.
- **#2 short response** now reported: `assess` takes the requested window and computes the
  shortfall at each end; `sweep` prints a `yrs` column and a `SHORT` flag and warns by
  name. Verified live: RDDT reports 2.11y and ARM 2.64y against a 5y request, both
  flagged, where both previously sat silently in a table headed "5y window". Cache writes
  are now atomic (temp plus rename).
- **#3 duplicated rows** detected explicitly, since set differences cannot see them, and
  `rows != expected_sessions` is now a failure when nothing else explains it.
- **#4 risk-free**: the fill is bounded to 7 days per print, `strict` guards a SHORT
  response as well as a missing one, levels outside [-1%, 25%] are rejected as scaling
  errors, and coverage is reported in the quality report. The docstring's understated
  approximation error (8bp claimed, 15.4bp at 4% and 25.3bp at the 5.4% peak, negative
  every day) is corrected.
- **#5 splits** are now genuinely cross-referenced, at any ratio, using a tolerance
  relative to `log(ratio)`. An absolute log tolerance was unusable: 0.15 around
  `-log(1.05)` spans -0.199 to +0.101 and swallows almost any trading day. Ratios down to
  1.05 are verifiable in both directions; below 1.03 the check declines explicitly, since a
  1% stock dividend is not separable from a 1% down day.
- **CLI now corroborates.** `backtest` and `sweep` previously called `assess` with no
  second source and still returned clean.

**Still open, and still before a multi-ticker batch.**

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

## WRITEUP.md: rewritten for WMT, all corrections applied

Rewritten 2026-07-29 for the graded ticker. Every one of the nine defects the review found
in the TSLA draft is addressed, and all 16 quoted numbers were re-verified against a fresh
run before committing (16/16).

What changed beyond the ticker:
- The markup comparison is now **like-for-like at the pre-specified 0.25 delta only**. The
  old draft paired that delta's baseline against the 0.40-delta cell, which was the argmax
  of the whole grid on both metrics. On WMT it happens not to matter, since the overlay beats
  buy and hold at every markup, but the comparison is stated correctly regardless.
- Sharpe is reported as **0.996, not 1.00**. It rounds to the target and does not clear it,
  and the gap between those two sentences is the assignment's actual question.
- The T-bill is quoted from its **own prints** (mean 3.62%, range 0.02% to 5.35%), not the
  log-transformed internal series.
- The TSLA leverage anecdote now says **4.4x exposure** and drops the "-240 against a share
  worth 222" pairing, which was two different days.
- `SWEEP_UNIVERSE` is a constant in the script, so "ten large caps" is checkable.
- Sortino is target downside deviation, and says so.
- **Calendar-year attribution is included**, because this is what killed the TSLA draft. WMT's
  edge is broad: it won 4 of 6 years, and excluding its worst year (2024, -16.6pp) the overlay
  still leads +62.42% to +46.13%. The opposite of the TSLA protective put, whose entire edge
  was 2022.
- The claim "every number is produced by a tested library" is narrowed to the performance
  numbers, since the assembly and prose arithmetic are not.
- Universals are gone. The real-IV limitation is stated as what was measured, not asserted.

## Batch: RUN 2026-07-29, and the result reframes everything

`scripts/watchlist_batch.py`, union of the three watchlists. 24 of 35 names usable, 19 with
the full five-year window, 10 strategies each. Paired per-ticker against buy-and-hold on the
same name over the same window, summarised by a sign test on the difference.

**No strategy beat buy-and-hold on average. Every one had a negative mean Sharpe difference.**

| strategy | beat on Sharpe | sign p | mean dSharpe | mean dReturn |
|---|---|---|---|---|
| protective put 5% | 7/19 | 0.359 | -0.039 | -60.7% |
| covered call 0.25d | **9/19** | **1.000** | -0.043 | -75.4% |
| bull call spread | 7/19 | 0.359 | -0.063 | -123.5% |
| collar 5/5 | 9/19 | 1.000 | -0.065 | -120.6% |
| wheel 0.25d | 7/19 | 0.359 | -0.073 | -112.0% |
| zero-cost collar | 9/19 | 1.000 | -0.090 | -126.8% |
| cash-secured put | 5/19 | 0.064 | -0.142 | -138.2% |
| long straddle | 4/19 | 0.019 | -0.360 | -157.0% |
| iron condor | 5/19 | 0.064 | -0.404 | -159.8% |
| short strangle | 4/17 | 0.049 | -0.427 | -191.2% |

Buy-and-hold's own mean Sharpe across those names: 0.414. The covered call beat its benchmark
on Sharpe in exactly **9 of 19**, a literal coin flip.

Three things this says, in order of importance.

1. **WMT is a favourable case, not a general finding.** Its Sharpe 1.00 against 0.80 is one
   cell out of 190. The graded writeup reports a 10-name cross-check at 4 of 10; this larger
   run says the covered call is a coin flip on risk-adjusted return and a large loser on total
   return across the watchlist. The writeup's claim is scoped to WMT and should stay there.
2. **The ordering is exactly what a zero variance risk premium predicts.** The more premium a
   structure sells, the worse it does: short strangle worst, then iron condor, then the
   cash-secured put. The *long*-premium straddle also loses, because it pays for a spread that
   does not exist either. Priced at realised vol every option trade is a fair bet, so all that
   remains is the give-up, and on a watchlist full of high-growth names (NVDA, AMD, META,
   MSTR, PANW, CRWD) capping the upside is expensive. The realised-vol limitation has stopped
   being a caveat and become a measured result.
3. **The straddle failing at p=0.019 is the control working.** A strategy set where nothing is
   permitted to lose would be a broken strategy set.

### The gate is now the binding problem

11 of 35 names were screened out, and most were rejected for REAL price moves rather than data
errors: BMNR +694.8%, AMC +95.2%, NWBO -58.2%, GME -33.8%, and NFLX for its genuine
2022-04-20 subscriber crash. The outlier check has no warning tier, so a legitimately violent
small cap cannot pass. This is round one's "hard failures with no override" item arriving as a
concrete 31% coverage loss, and it is now the highest-value fix: an acknowledged-exceptions
list or a warning tier, so a real move is disclosed rather than disqualifying the name.

Cosmetic but misleading: the message reads "move(s) over 40%" then prints -35.1%. Both are
correct (the threshold is on log returns, the display is simple) but the wording invites
exactly the wrong conclusion.

## Original batch scope, for reference

Union of all three watchlists (~35 equities), screened for optionability
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
