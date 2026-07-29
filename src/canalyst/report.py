"""Figures and tables. Matplotlib only, no seaborn, no style packages."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import Performance  # noqa: E402

FIGSIZE = (11, 6)
DPI = 130


def plot_price_and_buy_hold(
    close: pd.Series, ticker: str, path: Path, total_return: pd.Series | None = None
) -> Path:
    """Assignment task 2: the price history and what holding it returned.

    Two panels because they answer different questions. The top is the price in dollars,
    which is what the option strikes are struck against. The bottom is growth of one
    dollar, which is the buy-and-hold pattern itself and is scale-free.
    """
    fig, (ax_price, ax_growth) = plt.subplots(
        2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * 1.25), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    ax_price.plot(close.index, close.values, linewidth=1.1, color="#1f4e79")
    ax_price.set_ylabel(f"{ticker} close (USD)")
    ax_price.set_title(
        f"{ticker}: {close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d}  "
        f"(split-adjusted)"
    )
    ax_price.grid(alpha=0.25)

    growth = close / close.iloc[0]
    ax_growth.plot(
        growth.index, growth.values, linewidth=1.3, color="#1f4e79",
        label="buy and hold (price)",
    )
    if total_return is not None and not total_return.equals(close):
        tr_growth = total_return / total_return.iloc[0]
        ax_growth.plot(
            tr_growth.index, tr_growth.values, linewidth=1.1, linestyle="--",
            color="#2e7d32", label="buy and hold (total return)",
        )
    ax_growth.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax_growth.set_ylabel("growth of $1")
    ax_growth.set_xlabel("date")
    ax_growth.grid(alpha=0.25)
    ax_growth.legend(frameon=False)

    # Anchor the label to the axes, not to the final data point: an offset from the last
    # value lands on top of the line whenever the series ends mid-range.
    total = growth.iloc[-1] - 1.0
    ax_growth.text(
        0.015, 0.93,
        f"buy and hold over the window: {total:+.1%}",
        transform=ax_growth.transAxes,
        fontsize=10, color="#1f4e79", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="#c8d6e5"),
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_strategy_comparison(
    curves: dict[str, pd.Series],
    ticker: str,
    path: Path,
    title: str | None = None,
) -> Path:
    """Assignment task 3: the overlay against the benchmark, plus drawdowns.

    Drawdown gets its own panel because it is where a covered call earns its keep. Two
    curves ending at the same level are not the same investment if one of them fell
    twice as far along the way, and that difference is exactly what Sharpe prices.
    """
    fig, (ax_value, ax_dd) = plt.subplots(
        2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * 1.3), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )
    palette = ["#1f4e79", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00"]

    for (name, series), colour in zip(curves.items(), palette):
        growth = series / series.iloc[0]
        ax_value.plot(growth.index, growth.values, linewidth=1.3, label=name, color=colour)
        drawdown = series / series.cummax() - 1.0
        ax_dd.plot(drawdown.index, drawdown.values, linewidth=1.0, color=colour)

    ax_value.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax_value.set_ylabel("growth of $1")
    ax_value.set_title(title or f"{ticker}: option overlay versus buy and hold")
    ax_value.grid(alpha=0.25)
    ax_value.legend(frameon=False)

    ax_dd.set_ylabel("drawdown")
    ax_dd.set_xlabel("date")
    ax_dd.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_sensitivity(
    frame: pd.DataFrame, path: Path, title: str, ylabel: str
) -> Path:
    """Sensitivity grid, drawn as grouped bars. Never the headline result."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    frame.plot(kind="bar", ax=ax, width=0.8, edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def performance_markdown(results: dict[str, Performance]) -> str:
    """Headline statistics as a markdown table, ready to paste into a writeup."""
    header = (
        "| metric | " + " | ".join(results.keys()) + " |\n"
        "|---|" + "---|" * len(results) + "\n"
    )
    rows = [
        ("cumulative return", lambda p: f"{p.cumulative_return:+.2%}"),
        ("CAGR", lambda p: f"{p.cagr:+.2%}"),
        ("annualised volatility", lambda p: f"{p.annual_vol:.2%}"),
        ("Sharpe (daily)", lambda p: f"{p.sharpe_daily:.2f}"),
        ("Sharpe (monthly)", lambda p: f"{p.sharpe_monthly:.2f}"),
        ("Sortino", lambda p: f"{p.sortino:.2f}"),
        ("max drawdown", lambda p: f"{p.max_drawdown:.2%}"),
    ]
    body = "".join(
        f"| {label} | " + " | ".join(fmt(p) for p in results.values()) + " |\n"
        for label, fmt in rows
    )
    return header + body
