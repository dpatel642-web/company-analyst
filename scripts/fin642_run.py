#!/usr/bin/env python3
"""FIN642: buy-and-hold versus a rolling covered call.

Three tasks:
  1. five years of daily data for one ticker
  2. plot it and show the buy-and-hold pattern
  3. an option overlay, compared honestly against buy-and-hold

Run:  make run     or    .venv/bin/python scripts/fin642_run.py --ticker TSLA

Everything numeric here comes from the tested library in src/canalyst. This file only
wires it together and prints, so there is no analysis logic to get wrong twice.

PRE-SPECIFIED, before looking at any result:
  monthly third-Friday rolls, 0.25-delta short call, no fees, priced off 60-day
  realised volatility with the real 13-week T-bill as the risk-free rate.
The sensitivity grid at the end exists to show robustness. It is NOT how the headline
configuration was chosen: picking whichever cell scores best is how a backtest gets
overfitted into a result that does not survive contact with the future.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from canalyst.backtest import run_backtest  # noqa: E402
from canalyst.data.calendar import roll_schedule  # noqa: E402
from canalyst.data.prices import load_history  # noqa: E402
from canalyst.data.providers.nasdaq_p import NasdaqTailProvider  # noqa: E402
from canalyst.data.quality import assess  # noqa: E402
from canalyst.data.riskfree import risk_free_series  # noqa: E402
from canalyst.metrics import comparison_table, summarise  # noqa: E402
from canalyst.options.vol import apply_markup, close_to_close  # noqa: E402
from canalyst.report import (  # noqa: E402
    performance_markdown,
    plot_price_and_buy_hold,
    plot_sensitivity,
    plot_strategy_comparison,
)
from canalyst.strategies.buy_hold import BuyHold  # noqa: E402
from canalyst.strategies.covered_call import CoveredCall  # noqa: E402
from canalyst.strategies.protective_put import ProtectivePut  # noqa: E402

VOL_LOOKBACK = 60
TARGET_DELTA = 0.25
IV_MARKUP_GRID = [1.00, 1.10, 1.20]
DELTA_GRID = [0.15, 0.25, 0.40]
RULE = "=" * 78

#: The cross-ticker universe, recorded so the writeup's claim about it is reproducible.
#: A previous draft said "ten large caps" with no list anywhere in the repo, which meant a
#: reader could not tell 1-of-10 from the survivor of a wider search.
SWEEP_UNIVERSE = [
    "TSLA", "KO", "JNJ", "WMT", "COST", "MSFT", "AAPL", "SPY", "LLY", "PG",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="WMT")
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--delta", type=float, default=TARGET_DELTA)
    p.add_argument(
        "--iv-markup", type=float, default=1.00,
        help="pricing vol as a multiple of realised vol; 1.00 is the headline case",
    )
    p.add_argument("--outdir", default=str(REPO / "out"))
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    ticker = args.ticker.upper()

    end = pd.Timestamp.today().normalize()
    # Pad the download so the volatility lookback is warm on the first reported bar.
    raw_start = end - pd.DateOffset(years=args.years) - pd.Timedelta(days=150)

    # ---------------------------------------------------------------- 1. data
    print(RULE)
    print(f"TASK 1  five years of daily data for {ticker}")
    print(RULE)
    history = load_history(ticker, raw_start, end, use_cache=not args.no_cache)
    tail = NasdaqTailProvider().recent_closes(ticker)
    quality = assess(
        history, tail_closes=tail, tail_source="nasdaq",
        requested_start=raw_start, requested_end=end,
    )
    print(quality.render())
    if not quality.clean:
        print("\nSTOPPING: the data did not pass its integrity checks.")
        print("Performance numbers computed on this series would not be trustworthy.")
        return 1

    close_full = history.close
    sigma_full = close_to_close(close_full, window=VOL_LOOKBACK)

    # Report only the last N years, after the vol lookback is warm.
    window_start = end - pd.DateOffset(years=args.years)
    mask = (close_full.index >= window_start) & sigma_full.notna()
    close = close_full.loc[mask]
    realised_vol = sigma_full.loc[mask]
    dividends = history.dividends.reindex(close.index).fillna(0.0)
    total_return = history.total_return_close.reindex(close.index)

    rf = risk_free_series(close.index)
    schedule = roll_schedule(pd.DatetimeIndex(close.index))

    print(f"\nreporting window   {close.index[0]:%Y-%m-%d} .. {close.index[-1]:%Y-%m-%d}")
    print(f"trading days       {len(close)}")
    print(f"monthly rolls      {len(schedule)}")
    print(f"realised vol       {realised_vol.mean():.1%} mean, "
          f"{realised_vol.min():.1%} to {realised_vol.max():.1%}")
    print(f"risk-free rate     {rf.mean():.2%} mean "
          f"(the handout hardcodes 2.00% across this window)")
    print(f"dividends paid     {dividends.sum():.4f}")

    # ---------------------------------------------------------------- 2. the benchmark
    print()
    print(RULE)
    print("TASK 2  the buy-and-hold pattern")
    print(RULE)
    fig1 = plot_price_and_buy_hold(
        close, ticker, outdir / f"{ticker}_01_price_buy_hold.png", total_return
    )
    print(f"figure -> {fig1.relative_to(REPO)}")

    pricing_vol = apply_markup(realised_vol, args.iv_markup)
    common = dict(
        close=close, sigma=pricing_vol, rate=rf, schedule=schedule,
        dividends=dividends, ticker=ticker,
    )

    benchmark = run_backtest(BuyHold(), **common)
    benchmark.assert_identity()
    bh_perf = summarise(benchmark.value, rf, label="buy and hold")
    bh_perf.verify()
    print()
    print(bh_perf.render())

    # ---------------------------------------------------------------- 3. the overlay
    print()
    print(RULE)
    print("TASK 3  rolling covered call")
    print(RULE)
    print(f"pre-specified: monthly third-Friday rolls, {args.delta:.2f}-delta short "
          f"call, no fees,\n               pricing vol = realised x {args.iv_markup:.2f}")

    overlay = run_backtest(CoveredCall(target_delta=args.delta), **common)
    overlay.assert_identity()
    cc_perf = summarise(overlay.value, rf, label=f"covered call {args.delta:.2f}d")
    cc_perf.verify()

    put = run_backtest(ProtectivePut(strike_rule="moneyness", otm_pct=0.05), **common)
    put.assert_identity()
    put_perf = summarise(put.value, rf, label="protective put 5%")
    put_perf.verify()

    print()
    print(cc_perf.render())
    print(f"  rolls / assignments    {overlay.rolls} / {overlay.assignments}")
    print(f"  net premium collected  {overlay.net_premium:.2f} "
          f"({overlay.net_premium / close.iloc[0]:.1%} of the opening share price)")
    print()
    print(put_perf.render())

    print("\naccounting identity, worst residual across every bar:")
    for name, res in [
        ("buy and hold", benchmark), ("covered call", overlay), ("protective put", put)
    ]:
        print(f"  {name:<16} {res.max_abs_residual:.3e}")

    results = {
        "buy and hold": bh_perf,
        f"covered call {args.delta:.2f}d": cc_perf,
        "protective put 5%": put_perf,
    }
    print()
    print(RULE)
    print("COMPARISON")
    print(RULE)
    print(comparison_table(results).to_string(float_format=lambda v: f"{v:>10.4f}"))
    print()
    print(performance_markdown(results))

    fig2 = plot_strategy_comparison(
        {
            "buy and hold": benchmark.value,
            f"covered call {args.delta:.2f}d": overlay.value,
            "protective put 5%": put.value,
        },
        ticker,
        outdir / f"{ticker}_02_strategy_comparison.png",
        title=(
            f"{ticker}: covered call versus buy and hold "
            f"(pricing vol = realised x {args.iv_markup:.2f})"
        ),
    )
    print(f"figure -> {fig2.relative_to(REPO)}")

    # ------------------------------------------------------------- sensitivity, not selection
    print()
    print(RULE)
    print("SENSITIVITY  robustness only. The headline above was fixed in advance.")
    print(RULE)
    sharpe_grid = pd.DataFrame(index=[f"{d:.2f}d" for d in DELTA_GRID], dtype=float)
    return_grid = pd.DataFrame(index=[f"{d:.2f}d" for d in DELTA_GRID], dtype=float)
    for markup in IV_MARKUP_GRID:
        vol = apply_markup(realised_vol, markup)
        sharpes, returns = [], []
        for delta in DELTA_GRID:
            res = run_backtest(
                CoveredCall(target_delta=delta),
                close=close, sigma=vol, rate=rf, schedule=schedule,
                dividends=dividends, ticker=ticker,
            )
            res.assert_identity()
            perf = summarise(res.value, rf, label=f"{delta}")
            sharpes.append(perf.sharpe_daily)
            returns.append(perf.cumulative_return)
        sharpe_grid[f"IV x{markup:.2f}"] = sharpes
        return_grid[f"IV x{markup:.2f}"] = returns

    print("\nSharpe (daily), by short-call delta and pricing-vol markup:")
    print(sharpe_grid.to_string(float_format=lambda v: f"{v:>8.2f}"))
    print(f"\nbuy and hold Sharpe for reference: {bh_perf.sharpe_daily:.2f}")
    print("\ncumulative return, same grid:")
    print(return_grid.to_string(float_format=lambda v: f"{v:>9.2%}"))
    print(f"\nbuy and hold cumulative return for reference: "
          f"{bh_perf.cumulative_return:.2%}")

    # The only like-for-like markup comparison: the PRE-SPECIFIED delta, at each markup.
    # Quoting the grid's best cell against the pre-specified cell's baseline is how a
    # sensitivity gets laundered into a result, and an earlier draft of the writeup did
    # exactly that.
    row = f"{args.delta:.2f}d"
    print(f"\nLIKE-FOR-LIKE, at the pre-specified {args.delta:.2f} delta only:")
    for column in return_grid.columns:
        ret, shp = return_grid.loc[row, column], sharpe_grid.loc[row, column]
        verdict = "beats" if ret > bh_perf.cumulative_return else "loses to"
        print(f"  {column}: {ret:+7.2%}  Sharpe {shp:.2f}   -> {verdict} buy and hold")

    print("\nCALENDAR-YEAR ATTRIBUTION (is any edge broad, or one year?)")
    years = pd.DataFrame({
        "buy and hold": bh_perf.calendar_years,
        "covered call": cc_perf.calendar_years,
        "protective put": put_perf.calendar_years,
    })
    years["cc - bh"] = years["covered call"] - years["buy and hold"]
    print(years.to_string(float_format=lambda v: f"{v:+8.2%}"))
    beat = int((years["cc - bh"] > 0).sum())
    print(f"\n  covered call beat buy and hold in {beat} of {len(years)} calendar years")
    for drop in years.index:
        ex = lambda s: (1 + s.drop(drop)).prod() - 1  # noqa: E731
        if abs(years.loc[drop, "cc - bh"]) > 0.05:
            print(f"  excluding {drop}: buy and hold {ex(years['buy and hold']):+.2%} "
                  f"vs covered call {ex(years['covered call']):+.2%}")

    fig3 = plot_sensitivity(
        sharpe_grid, outdir / f"{ticker}_03_sensitivity.png",
        title=f"{ticker} covered call: Sharpe by delta and pricing-vol markup",
        ylabel="Sharpe (daily)",
    )
    print(f"\nfigure -> {fig3.relative_to(REPO)}")

    print()
    print(RULE)
    print("READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
    print(RULE)
    print(
        "Options are priced off realised volatility, because no free source carries a\n"
        "five-year history of option chains. Implied volatility trades above realised\n"
        "most of the time, and that spread is the entire reason writing calls pays. At\n"
        "IV x1.00 the variance risk premium is therefore set to zero and the overlay is\n"
        "close to a fair bet, which understates it. The markup column shows what a real\n"
        "premium would have been worth on the same price path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
