# company-analyst

Company analysis toolkit: price history, option overlays, backtests.

**Read-only by construction.** There is no broker client and no order function. This is
the analysis half of what used to be a trading agent, and the separation is structural
rather than a policy note.

## Setup

    make venv
    make test

Python 3.14 (brew). All dependencies ship cp314 wheels; scipy is deliberately absent,
since `statistics.NormalDist` covers every normal CDF and quantile Black-Scholes needs.

## Known sharp edges

Documented because each one silently produces plausible wrong numbers, and each has a
test pinning it.

- **Prices are retroactively split-adjusted, always.** Yahoo returns split-adjusted
  history even with `auto_adjust=False`; true as-traded levels are not available from
  any free source. TSLA closed near $891 on 2022-08-24 and near $296 the next day after
  a 3-for-1, and the series shows no discontinuity. This is the right basis rather than
  a defect, because the OCC adjusts open option contracts the same way, but it only
  works if the adjustment is *consistent*. `data/quality.py` verifies that.
- **Two close series, not interchangeable.** `frame["Close"]` is split-adjusted and
  dividend-unadjusted, and is what strikes and assignment use. `total_return_close` is
  also dividend-adjusted, and is the buy-and-hold benchmark. Striking options off the
  total-return series moves every strike by the cumulative dividend yield.
- **yfinance returns bars for sessions that have not closed.** Verified live on
  2026-07-29, which returned a row while the last completed session was 07-28. Left in,
  that partial bar becomes the final close and every terminal number inherits it.
  `drop_incomplete_bar` removes it.
- **The third Friday is not always a trading day.** 2025-04-18 was both April's third
  Friday and Good Friday, so that month's expiry was Thursday 04-17. Expiries align
  backwards to the preceding session, never forwards.
- **Options are priced off realised volatility, not implied.** No free source carries
  five years of option chains. Implied trades above realised on average, so this
  *understates* premium collected and the base case is conservative. `iv_markup` measures
  the sensitivity; a marked-up number is never the headline.
- **Full-window source independence is not verified.** Stooq is bot-gated as of
  2026-07-29 and Nasdaq serves only about ten sessions, so no free no-key source covers
  five years. The recent tail is corroborated externally; the body is corroborated only
  internally. Reports say so rather than implying dual-source agreement.

## Data sources

| Source | Use | Terms |
|---|---|---|
| Yahoo (yfinance) | Daily OHLCV, dividends, splits | Undocumented, behind a provider interface |
| api.nasdaq.com | Independent closes, last ~10 sessions | Undocumented, behind a provider interface |
| Yahoo `^IRX` | 13-week T-bill, the risk-free series | Undocumented |

**Not WRDS.** Its terms restrict data to academic and non-commercial research; personal
analysis violates them. Nothing here touches it.
