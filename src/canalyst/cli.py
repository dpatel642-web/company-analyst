"""Command line entry points.

    canalyst backtest --ticker TSLA
    canalyst sweep --tickers TSLA,PG,MSFT --years 5

`sweep` exists because a single name cannot tell you whether a result is a property of
the strategy or a property of that stock. Running the same pre-specified overlay across
several names separates the two.

It is a diagnostic, not a stock picker. Reading the sweep and then declaring the
best-scoring name the "chosen" ticker would be selection on the outcome, which is how a
backtest gets overfitted. Choose the name first, for a reason outside the data.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtest import run_backtest
from .data.calendar import roll_schedule
from .data.prices import load_history
from .data.providers.nasdaq_p import NasdaqTailProvider
from .data.quality import assess
from .data.riskfree import risk_free_series
from .metrics import Performance, summarise
from .options.vol import apply_markup, close_to_close
from .strategies.buy_hold import BuyHold
from .strategies.covered_call import CoveredCall

VOL_LOOKBACK = 60


def _prepare(ticker: str, years: float, use_cache: bool = True):
    """Load, verify, and window one ticker. Raises if the data is not trustworthy."""
    end = pd.Timestamp.today().normalize()
    raw_start = end - pd.DateOffset(years=years) - pd.Timedelta(days=150)

    history = load_history(ticker, raw_start, end, use_cache=use_cache)
    close_full = history.close
    sigma_full = close_to_close(close_full, window=VOL_LOOKBACK)
    window_start = end - pd.DateOffset(years=years)
    mask = (close_full.index >= window_start) & sigma_full.notna()

    close = close_full.loc[mask]
    if len(close) < 200:
        raise RuntimeError(f"{ticker}: only {len(close)} usable bars")

    rf, rf_had_print = risk_free_series(close.index, return_coverage=True)

    # Corroborate the recent tail. The CLI used to call assess() with no second source,
    # so the integrity gate ran with zero external corroboration and still returned clean.
    tail = NasdaqTailProvider().recent_closes(ticker)
    quality = assess(
        history,
        tail_closes=tail,
        tail_source="nasdaq",
        requested_start=raw_start,
        requested_end=end,
        rf=rf,
        rf_had_print=rf_had_print,
    )
    if not quality.clean:
        raise RuntimeError(
            f"{ticker}: data failed integrity checks: {'; '.join(quality.failures)}"
        )

    return (
        close,
        sigma_full.loc[mask],
        history.dividends.reindex(close.index).fillna(0.0),
        rf,
        roll_schedule(pd.DatetimeIndex(close.index)),
        quality,
    )


def _evaluate(
    ticker: str, years: float, delta: float, markup: float, use_cache: bool = True
) -> dict:
    close, realised, dividends, rf, schedule, quality = _prepare(ticker, years, use_cache)
    common = dict(
        close=close, sigma=apply_markup(realised, markup), rate=rf,
        schedule=schedule, dividends=dividends, ticker=ticker,
    )

    benchmark = run_backtest(BuyHold(), **common)
    overlay = run_backtest(CoveredCall(target_delta=delta), **common)
    benchmark.assert_identity()
    overlay.assert_identity()

    bh: Performance = summarise(benchmark.value, rf, label="bh")
    cc: Performance = summarise(overlay.value, rf, label="cc")
    bh.verify()
    cc.verify()

    return {
        "ticker": ticker,
        "from": close.index[0].date(),
        "to": close.index[-1].date(),
        "years": round(bh.elapsed_years, 2),
        "short": quality.missing_head_sessions + quality.missing_tail_sessions > 5,
        "vol": bh.annual_vol,
        "bh_return": bh.cumulative_return,
        "bh_sharpe": bh.sharpe_daily,
        "cc_return": cc.cumulative_return,
        "cc_sharpe": cc.sharpe_daily,
        "cc_vol": cc.annual_vol,
        "beats_bh": cc.cumulative_return > bh.cumulative_return,
        "sharpe_over_1": cc.sharpe_daily >= 1.0,
        "assignments": overlay.assignments,
        "rolls": overlay.rolls,
    }


def cmd_backtest(args: argparse.Namespace) -> int:
    close, realised, dividends, rf, schedule, quality = _prepare(
        args.ticker.upper(), args.years, not args.no_cache
    )
    print(quality.render())
    print()

    common = dict(
        close=close, sigma=apply_markup(realised, args.iv_markup), rate=rf,
        schedule=schedule, dividends=dividends, ticker=args.ticker.upper(),
    )
    benchmark = run_backtest(BuyHold(), **common)
    overlay = run_backtest(CoveredCall(target_delta=args.delta), **common)
    benchmark.assert_identity()
    overlay.assert_identity()

    for label, result in [("buy and hold", benchmark), ("covered call", overlay)]:
        perf = summarise(result.value, rf, label=label)
        perf.verify()
        print(perf.render())
        print()
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    rows, failures = [], []
    for ticker in tickers:
        try:
            rows.append(
                _evaluate(ticker, args.years, args.delta, args.iv_markup, not args.no_cache)
            )
            print(f"  ok    {ticker}", file=sys.stderr)
        except Exception as exc:  # a bad ticker must not kill the sweep
            failures.append((ticker, str(exc)))
            print(f"  FAIL  {ticker}: {exc}", file=sys.stderr)

    if not rows:
        print("no ticker produced a usable result")
        return 1

    frame = pd.DataFrame(rows).set_index("ticker")
    print()
    print(
        f"Covered call {args.delta:.2f}-delta, monthly rolls, pricing vol = realised "
        f"x {args.iv_markup:.2f}, {args.years:g}y window"
    )
    print("=" * 96)
    display = frame[
        ["years", "short", "vol", "cc_vol", "bh_return", "cc_return", "bh_sharpe",
         "cc_sharpe", "beats_bh", "sharpe_over_1"]
    ].rename(
        columns={
            "years": "yrs", "short": "SHORT",
            "vol": "BH vol", "cc_vol": "CC vol", "bh_return": "BH ret",
            "cc_return": "CC ret", "bh_sharpe": "BH Sharpe", "cc_sharpe": "CC Sharpe",
            "beats_bh": "CC>BH", "sharpe_over_1": "Sharpe>=1",
        }
    )
    with pd.option_context("display.width", 200):
        print(
            display.to_string(
                formatters={
                    "BH vol": "{:.1%}".format, "CC vol": "{:.1%}".format,
                    "BH ret": "{:+.1%}".format, "CC ret": "{:+.1%}".format,
                    "BH Sharpe": "{:.2f}".format, "CC Sharpe": "{:.2f}".format,
                }
            )
        )
    print()
    short = frame[frame["short"]]
    if len(short):
        print(
            f"WARNING: {len(short)} ticker(s) returned materially less than the "
            f"requested window: {', '.join(short.index)}. Their 'yrs' column is the "
            "real span; do not read them as full-window results."
        )
    print(f"beats buy and hold : {int(frame['beats_bh'].sum())} of {len(frame)}")
    print(f"Sharpe >= 1        : {int(frame['sharpe_over_1'].sum())} of {len(frame)}")
    if failures:
        print(f"failed             : {', '.join(t for t, _ in failures)}")
    print()
    print(
        "Sharpe is (excess return / volatility). A high-volatility name needs an\n"
        "implausibly large return to clear 1.00 no matter which overlay is applied, so a\n"
        "low Sharpe here is usually a fact about the stock rather than the strategy."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canalyst", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--years", type=float, default=5.0)
    common.add_argument("--delta", type=float, default=0.25)
    common.add_argument("--iv-markup", type=float, default=1.00)
    common.add_argument("--no-cache", action="store_true")

    p_back = sub.add_parser("backtest", parents=[common], help="one ticker in detail")
    p_back.add_argument("--ticker", required=True)
    p_back.set_defaults(func=cmd_backtest)

    p_sweep = sub.add_parser("sweep", parents=[common], help="the same overlay, many names")
    p_sweep.add_argument("--tickers", required=True, help="comma separated")
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
