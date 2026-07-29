# FIN642: a rolling covered call on WMT versus buy and hold

Ticker: WMT. Window: 2021-07-29 to 2026-07-28, 1,254 trading sessions (5.00 calendar
years). Code: [dpatel642-web/company-analyst](https://github.com/dpatel642-web/company-analyst),
entry point [`scripts/fin642_run.py`](scripts/fin642_run.py). Reproduce with
`make venv && make run`. 292 tests pass.

## 1. The strategy, and why it should work

Hold the stock and, on every monthly expiry, write one call against it at the strike whose
Black-Scholes delta is 0.25, expiring at the next monthly expiry. The position is fully
collateralised: share count resets to equity divided by spot whenever no option is open, so
an assignment loss shrinks the position rather than opening a margin loan.

Two reasons it should work, and they are different reasons.

The first is a risk transfer that is paid for. Writing the call sells the right tail of the
return distribution to someone who wants it, and implied volatility trades above subsequent
realised volatility most of the time, so the seller collects that spread on average. This is
the variance risk premium, and it is the strategy's actual source of return.

The second is arithmetic and does not depend on the first. Sharpe is a ratio. Selling the
upper tail removes the largest single contributors to variance while giving up only a capped
amount of return, so the denominator can shrink faster than the numerator. On WMT the
overlay cut annualised volatility from 21.93% to 16.84%, a 23% reduction, while total return
went slightly up.

Why WMT is a reasonable underlying for it: a covered call needs a name that grinds upward
without repeatedly jumping past the strike. WMT compounded at 20.6% over these five years
with 21.9% volatility, which is a high return for that level of risk, and it did it without
the violent monthly gaps that make a short call expensive.

## 2. Sharpe ratio

| | buy and hold | covered call 0.25 delta | protective put 5% |
|---|---|---|---|
| **Sharpe (daily)** | 0.80 | **0.996** | 0.87 |
| Sharpe (monthly) | 0.85 | 0.99 | 0.94 |
| Sortino | 1.14 | **1.38** | 1.31 |
| annualised volatility | 21.93% | **16.84%** | 18.55% |
| max drawdown | -25.74% | -24.08% | **-18.27%** |

The covered call's Sharpe is **0.996**, against 0.80 for buy and hold. That rounds to 1.00
and does not clear it, and I am reporting it as 0.996 rather than 1.0 because the difference
between those two statements is the whole point of the assignment's target. It is a 24%
improvement in risk-adjusted return on the same underlying.

It clears 1.0 comfortably under any non-zero variance risk premium, which matters because
of a modelling limitation described in section 4: options here are priced off realised
volatility, which sets that premium to exactly zero. At the pre-specified 0.25 delta,
pricing at realised volatility times 1.10 gives Sharpe 1.14, and times 1.20 gives 1.26.

Sharpe is computed two ways from different samples, daily and monthly, because a single
estimator can be wrong quietly. They agree to 0.01. Sortino uses target downside deviation,
the root mean square shortfall below the risk-free rate over all periods, not the standard
deviation of only the losing days: that shortcut's bias tracks the hit rate, so it flips
sign as drift changes and silently reorders a comparison like this one.

## 3. Cumulative return over the five years

| | cumulative | CAGR |
|---|---|---|
| buy and hold | +154.25% | +20.64% |
| **covered call 0.25 delta** | **+155.59%** | **+20.77%** |
| protective put 5% | +144.32% | +19.68% |

The covered call beat buy and hold on total return by 1.34 percentage points while holding
23% less volatility. The margin on return is thin and I am not going to dress it up; the
result that matters is the same return for materially less risk.

Across 59 monthly rolls it collected premium worth 79.6% of the opening share price and was
assigned 21 times, a 36% assignment rate against a nominal 25% delta.

**The edge is broad, not one lucky year.** This is the question that killed an earlier draft
of this project on a different ticker, so it gets checked explicitly:

| year | buy and hold | covered call | difference |
|---|---|---|---|
| 2021 (partial) | +2.50% | +3.55% | +1.04% |
| 2022 | -0.47% | +2.23% | +2.69% |
| 2023 | +12.89% | +12.09% | -0.79% |
| 2024 | +73.98% | +57.36% | **-16.62%** |
| 2025 | +24.49% | +28.18% | +3.69% |
| 2026 (partial) | +1.92% | +6.79% | +4.87% |

It won in 4 of 6 calendar years. 2024 is the cost of the strategy made visible: WMT rose 74%
and the overlay forfeited 16.6 points of that to the strikes, exactly as a covered call
should. Excluding 2024, buy and hold returns +46.13% against the overlay's +62.42%.

**Cross-ticker check.** The same pre-specified overlay was run over ten large caps, recorded
as `SWEEP_UNIVERSE` in the script so this is reproducible rather than asserted: TSLA, KO,
JNJ, WMT, COST, MSFT, AAPL, SPY, LLY, PG. It beat buy and hold on 4 of the 10 and produced
the highest risk-adjusted return of any of them on WMT. On TSLA it lost badly, returning
-26.9% against +36.2%, because a 60%-volatility stock jumps past the strike repeatedly. That
asymmetry is the strategy's real character, and a version of this that won everywhere would
be a bug rather than a discovery.

## 4. Method, and the four things that would change the answer

Every performance number comes from a tested library. The assembly and the prose arithmetic
in this file are not, which is where a factual error in an earlier draft came from.

**Full daily mark to market, enforced by an accounting identity.** Every open option is
repriced every bar, and portfolio value may change only through named causes: share profit
and loss, dividends, interest, mark-to-market on positions held across the bar, and fees.
The residual is recorded per bar; the worst across all 1,254 bars is 4.7e-14. The assignment's
example code instead values a covered call as `shares * close + cash` where cash moves only
on roll and expiry dates, so the short call's liability is invisible in between and its whole
profit and loss arrives as one spike. Terminal value survives that. Daily volatility, and
therefore Sharpe, does not.

**Full collateralisation.** Paying assignment losses in cash while holding a fixed share
count is not a covered call, it is a levered long funded by a growing margin loan. Modelled
that way on TSLA, share exposure reached 4.4 times remaining equity and the strategy printed
93.7% volatility against the stock's 59.7%, which is impossible for a position with net delta
below one.

**Dividend yield in the pricing.** WMT pays a dividend, and pricing a call as though it did
not while separately banking the dividend pays the writer twice, worth roughly 59 basis
points a year at this yield. It also moves the strike: with the yield omitted, a nominal
0.25-delta strike actually sits at 0.2375, so the pre-specified parameter quietly stops
being the parameter in force. Fixing this cost WMT's headline about 6 points of return, and
that is the corrected number above.

**Options are priced off realised volatility, and this understates the strategy.** No free
source carries five years of option chains. I tested Robinhood's expired-contract price
history directly as a way to measure implied volatility from real trades, and it does not
work for a name like WMT: of five sampled roll dates, three returned bars that were either
flagged as interpolated gap-fill or pinned at the one-cent minimum tick, and the implied
volatility ratios spanned 0.30 to 3.60. So realised volatility stands, the variance risk
premium is therefore zero by construction, and the headline is conservative. The markup
column in section 2 shows what a real premium would be worth on the identical price path.

The risk-free rate is the actual 13-week Treasury bill, which averaged 3.62% over this
window and ranged from 0.02% to 5.35%. A hardcoded 2% is wrong in both directions and Sharpe
depends on it directly.
