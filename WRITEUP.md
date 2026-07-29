# FIN642: an option overlay on TSLA versus buy and hold

Ticker: TSLA. Window: 2021-07-29 to 2026-07-28, 1,254 trading days (4.97 years).
Code: [`company-analyst`](https://github.com/dpatel642-web/company-analyst), entry point
[`scripts/fin642_run.py`](scripts/fin642_run.py). Reproduce with `make venv && make run`.

## 1. The strategy, and why it should work

The overlay is a **rolling protective put**: hold the stock, and on every monthly
expiry buy a put 5% below spot expiring at the next monthly expiry. I also ran a rolling
covered call, and reporting both is the point of this writeup.

A protective put buys the left tail. It costs premium every month, which is a permanent
drag on return, and in exchange it converts an unbounded drawdown into a capped one. The
reason it should work specifically here is a property of the underlying rather than of the
option: TSLA realised 59.7% annualised volatility over these five years and fell 73.6%
peak to trough. On a distribution that wide, the payoff from owning downside convexity
through one crash can exceed several years of premium outlay. That is what happened.

The covered call is the mirror trade. It sells the right tail, collects premium, and
forfeits every rally past the strike. It should work when a stock is choppy and trendless,
and it should fail when a stock delivers a small number of very large up months. TSLA is
the second case, so the covered call lost, and I report that rather than tuning around it.

## 2. Sharpe ratio

| | buy and hold | protective put 5% | covered call 0.25 delta |
|---|---|---|---|
| Sharpe (daily) | 0.34 | **0.42** | 0.01 |
| Sharpe (monthly) | 0.33 | 0.41 | 0.00 |
| annualised volatility | 59.7% | 44.0% | 45.5% |
| max drawdown | -73.6% | **-50.3%** | -63.3% |

The protective put improves Sharpe from 0.34 to 0.42 and cuts the worst drawdown by
23 percentage points. **It does not reach 1.0, and on this ticker nothing does.** Sharpe is
excess return divided by volatility, so at 44% volatility a Sharpe of 1.0 requires roughly
a 47% annualised excess return. TSLA compounded at 6.4% a year over this window. The
shortfall is a fact about the stock, not a tuning problem, and no overlay closes it.

For contrast, the same pre-specified covered call run across ten large caps clears the bar
on WMT: Sharpe 1.02 against buy and hold's 0.80, with a higher cumulative return
(+161.3% against +148.2%). The strategy is capable of Sharpe above 1. TSLA is not.

## 3. Cumulative return over the five years

| | cumulative | CAGR |
|---|---|---|
| buy and hold | +36.17% | +6.41% |
| protective put 5% | **+87.02%** | +13.42% |
| covered call 0.25 delta | -26.94% | -6.12% |

The protective put beats buy and hold by 51 percentage points while holding less risk on
every measure. The covered call collected 112% of the opening share price in premium
across 59 rolls and still lost, because 15 assignments in TSLA's largest up months cost
more than the premium taken in.

## 4. Method, and the four things that change the answer

Every number is produced by a tested library, not by the script. 198 tests pass.

**Full daily mark to market.** Every open option is repriced every day. The assignment's
example code values a covered call as `shares * close + cash`, where cash moves only on
roll and expiry days, so the short call's liability is invisible in between and its whole
profit and loss arrives as one spike at expiry. Terminal value survives that; daily
volatility, and therefore Sharpe, does not. The engine enforces a daily accounting
identity instead, so value can only change through share profit and loss, dividends,
interest, position marks, and fees. The worst residual across all 1,254 bars is 2.2e-13.

**Full collateralisation.** Paying assignment losses in cash while holding a fixed share
count is not a covered call, it is a levered long funded by a growing margin loan. Modelled
that way on TSLA, cash reached -240 against a share worth 222, and the strategy printed
93.7% volatility against the stock's 59.7%, which is impossible for a position with net
delta below one. Share count is therefore reset to equity divided by spot whenever no
option is open, which is how CBOE's BXM buy-write index is constructed.

**A real risk-free rate.** The 13-week Treasury bill averaged 3.54% over this window and
ranged from 0.02% to 5.21%. A hardcoded 2% is wrong in both directions, and Sharpe depends
on it directly.

**Options are priced off realised volatility, and this understates the covered call.**
No free source carries five years of option chains. Implied volatility trades above
subsequent realised volatility most of the time, and that spread is the entire reason
writing options pays. Pricing at realised volatility sets it to zero, which makes writing
a call close to a fair bet. Marking premiums up 20% toward where implied actually trades
turns the TSLA covered call from -26.9% into +63.3% and lifts its Sharpe from 0.01 to 0.36.
That is a sensitivity, not the headline, and the pre-specified configuration was fixed
before any result was seen.
