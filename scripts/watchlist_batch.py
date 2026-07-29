#!/usr/bin/env python3
"""Every strategy against every name on the watchlist, with paired statistics.

WHY PAIRED, AND WHY THAT IS THE WHOLE POINT

"The covered call got Sharpe 1.00 on WMT" is not evidence that covered calls work. With
eleven strategies and thirty-five names there are 385 cells, and the best of 385 cells is
going to look good whatever the underlying truth is. That is the same selection problem the
sensitivity grid has, one dimension larger.

The question a batch can answer that a single name cannot is whether a strategy beats its own
benchmark *consistently*, across names chosen before the results were seen. So every strategy
is scored on the per-ticker DIFFERENCE against buy-and-hold on the same ticker over the same
window, and summarised by how often that difference is positive plus a sign test on it.

Pairing matters because the tickers differ enormously: TSLA at 60% volatility and KO at 17%
have nothing to say to each other in levels, but "did the overlay beat holding this same
stock" is comparable across both.

The sign test is deliberately weak. It assumes only that, under the null of no effect, a
strategy is equally likely to beat or lose to its benchmark on each name. It does not assume
normal returns, equal variances, or independence across names, and the last of those is
false anyway since these names share market factors. A stronger test would need an
assumption this data cannot support, so the weak one is the honest one, and its p-values
should be read as indicative rather than exact.

Universe: the union of three watchlists, screened for whether an option overlay is even
meaningful. Exclusions are reported by name, never dropped silently.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from canalyst.backtest import run_backtest  # noqa: E402
from canalyst.cli import _prepare  # noqa: E402
from canalyst.metrics import summarise  # noqa: E402
from canalyst.strategies.buy_hold import BuyHold  # noqa: E402
from canalyst.strategies.cash_secured_put import CashSecuredPut  # noqa: E402
from canalyst.strategies.collar import Collar, ZeroCostCollar  # noqa: E402
from canalyst.strategies.covered_call import CoveredCall  # noqa: E402
from canalyst.strategies.premium_selling import IronCondor, ShortStrangle  # noqa: E402
from canalyst.strategies.protective_put import ProtectivePut  # noqa: E402
from canalyst.strategies.spreads import VerticalSpread  # noqa: E402
from canalyst.strategies.straddle import LongStraddle  # noqa: E402
from canalyst.strategies.wheel import Wheel  # noqa: E402

#: Union of the edgar-screener YAML (21), the Robinhood "Watchlist" (13) and "My First
#: List" (14), equities only. Recorded here so the universe is checkable rather than asserted.
WATCHLIST_UNION = [
    "NWBO", "TSLA", "BMNR", "ABNB", "AMZN", "ZM", "META", "COIN", "GME", "NVDA",
    "PANW", "CRWD", "MSTR", "SLNH", "IREN", "SNDK", "IBIT", "RVI", "TSLL", "LOWLF",
    "SPCX", "DRAM", "MU", "AAPL", "LCID", "RIVN", "MSFT", "GOOGL", "F", "AMD",
    "AMC", "DIS", "WMT", "NFLX", "KO",
]

#: Excluded before any data is fetched, with the reason. Crypto pairs are not securities and
#: have no listed equity options in this framework.
EXCLUDED_UP_FRONT = {
    "BTC-USD": "crypto pair, not a security",
    "SOL": "crypto pair, not a security",
}


def strategies():
    """Fresh instances every ticker: several carry state (the wheel, collars, share counts)."""
    return [
        CoveredCall(target_delta=0.25),
        CashSecuredPut(target_delta=0.25),
        ProtectivePut(strike_rule="moneyness", otm_pct=0.05),
        Collar(call_otm=0.05, put_otm=0.05),
        ZeroCostCollar(call_otm=0.05),
        Wheel(target_delta=0.25),
        ShortStrangle(call_otm=0.10, put_otm=0.10),
        IronCondor(call_otm=0.10, put_otm=0.10, wing_width=0.05),
        VerticalSpread("bull_call", 0.00, 0.10),
        LongStraddle(),
    ]


def sign_test(differences: list[float]) -> tuple[int, int, float]:
    """Two-sided sign test. Returns (wins, n, p).

    Exact binomial, no normal approximation, because n is around thirty and the
    approximation is poor in the tails that matter.
    """
    non_zero = [d for d in differences if d != 0.0]
    n = len(non_zero)
    wins = sum(1 for d in non_zero if d > 0)
    if n == 0:
        return 0, 0, float("nan")
    # P(X >= max(wins, n-wins)) * 2 under Binomial(n, 0.5)
    extreme = max(wins, n - wins)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return wins, n, min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--tickers", default=None, help="override the universe")
    parser.add_argument("--out", default=str(REPO / "out" / "watchlist_batch.csv"))
    args = parser.parse_args()

    universe = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers
        else WATCHLIST_UNION
    )

    print("Excluded before fetching:")
    for ticker, reason in EXCLUDED_UP_FRONT.items():
        print(f"  {ticker:<8} {reason}")

    rows: list[dict] = []
    screened_out: dict[str, str] = {}

    for ticker in universe:
        try:
            close, rv, div, rf, sched, quality = _prepare(ticker, args.years)
        except Exception as exc:
            screened_out[ticker] = str(exc)[:110]
            print(f"  SKIP  {ticker}: {str(exc)[:90]}", file=sys.stderr)
            continue

        short = quality.missing_head_sessions + quality.missing_tail_sessions > 5
        common = dict(
            close=close, sigma=rv, rate=rf, schedule=sched, dividends=div, ticker=ticker
        )
        bench = run_backtest(BuyHold(), **common)
        bench.assert_identity()
        bh = summarise(bench.value, rf, label="bh")

        for strategy in strategies():
            try:
                result = run_backtest(strategy, **common)
                result.assert_identity()
                perf = summarise(result.value, rf, label=strategy.name)
                perf.verify()
            except Exception as exc:
                print(f"  fail  {ticker}/{strategy.name}: {exc}", file=sys.stderr)
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy.name,
                    "years": round(bh.elapsed_years, 2),
                    "short_window": short,
                    "bh_return": bh.cumulative_return,
                    "return": perf.cumulative_return,
                    "return_diff": perf.cumulative_return - bh.cumulative_return,
                    "bh_sharpe": bh.sharpe_daily,
                    "sharpe": perf.sharpe_daily,
                    "sharpe_diff": perf.sharpe_daily - bh.sharpe_daily,
                    "bh_vol": bh.annual_vol,
                    "vol": perf.annual_vol,
                    "max_dd": perf.max_drawdown,
                    "bh_max_dd": bh.max_drawdown,
                }
            )
        print(f"  ok    {ticker} ({bh.elapsed_years:.2f}y)", file=sys.stderr)

    if not rows:
        print("\nno usable results")
        return 1

    frame = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"\nscreened out ({len(screened_out)}):")
    for ticker, reason in screened_out.items():
        print(f"  {ticker:<8} {reason}")
    usable = sorted(frame["ticker"].unique())
    full = sorted(frame.loc[~frame["short_window"], "ticker"].unique())
    print(f"\nusable: {len(usable)} tickers, of which {len(full)} have the full window")
    print(f"  full window : {', '.join(full)}")
    partial = sorted(set(usable) - set(full))
    if partial:
        print(f"  SHORT window: {', '.join(partial)} (excluded from the paired tests)")

    # Paired statistics on the full-window names only: a 2-year name and a 5-year name are
    # not comparable observations of the same question.
    paired = frame.loc[~frame["short_window"]]
    print("\n" + "=" * 92)
    print(f"PAIRED versus buy-and-hold, per ticker, {len(full)} full-window names")
    print("=" * 92)
    print(f"{'strategy':<28}{'beat':>7}{'sign p':>9}{'mean dSharpe':>14}"
          f"{'mean dReturn':>14}{'mean vol':>10}")
    summary = []
    for name, group in paired.groupby("strategy"):
        wins, n, p = sign_test(list(group["sharpe_diff"]))
        summary.append(
            {
                "strategy": name,
                "beat_sharpe": f"{wins}/{n}",
                "p": p,
                "mean_sharpe_diff": group["sharpe_diff"].mean(),
                "mean_return_diff": group["return_diff"].mean(),
                "mean_vol": group["vol"].mean(),
            }
        )
    for row in sorted(summary, key=lambda r: -r["mean_sharpe_diff"]):
        print(f"{row['strategy']:<28}{row['beat_sharpe']:>7}{row['p']:>9.3f}"
              f"{row['mean_sharpe_diff']:>+14.3f}{row['mean_return_diff']:>+14.1%}"
              f"{row['mean_vol']:>10.1%}")

    print(f"\nbuy-and-hold mean Sharpe across those names: "
          f"{paired.groupby('ticker')['bh_sharpe'].first().mean():.3f}")
    print(f"\nfull results -> {out_path.relative_to(REPO)}")
    print(
        "\nRead the sign-test p-values as indicative. These names share market factors, so\n"
        "the observations are not independent, and no test on thirty-odd correlated series\n"
        "supports a precise p-value. A strategy that beats its benchmark on most names is\n"
        "evidence; a single winning cell out of several hundred is not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
