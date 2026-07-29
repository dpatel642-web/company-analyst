"""Performance statistics, with an honest account of which checks actually check anything.

House rule is that a number worth acting on gets verified by a second path. That rule is
easy to satisfy in appearance and hard to satisfy in substance, so this module labels
which of its checks are real.

REAL: Sharpe from daily and from monthly sampling. Different estimators over different
samples, so a disagreement is informative. (They coincide exactly only under iid
returns, which is an assumption, not a fact.)

REAL, and living in backtest.py rather than here: the daily accounting identity. That is
the check with actual detection power over a corrupted equity curve.

NOT REAL, and previously mislabelled here as independent evidence:
`cumulative_return_check`. It computes `expm1(sum(log1p(pct_change)))`, and
`log1p(v_i/v_{i-1} - 1) == log(v_i) - log(v_{i-1})`, so the sum **telescopes** to
`log(v_n/v_0)` and the result is algebraically identical to `v_n/v_0 - 1`. Agreement to
1e-16 is float rounding on a telescoping sum, not corroboration. Verified to have zero
detection power: inject a fabricated 50% jump mid-curve and it still agrees to 3e-15.
It is retained purely as a float-stability tripwire and is documented as such. No metric
computed *from* an equity curve can detect that the curve itself is wrong; that has to be
caught upstream.

On Sharpe specifically. The textbook definition is the mean excess return over its
standard deviation, annualised by the square root of the sampling frequency. The handout
instead divides a geometrically-compounded annual return by an arithmetically-annualised
volatility, which mixes two conventions. That is not a rounding difference: on a
high-volatility name the two disagree materially, because compounding drags the
geometric return well below the arithmetic mean. Both are reported here, labelled.

On Sortino. The denominator is target downside deviation,
`sqrt(mean over ALL periods of min(r - MAR, 0)^2)` with MAR = the risk-free rate. It is
NOT the standard deviation of the negative subset, which is a common shortcut and a wrong
one: dividing by `n_neg` instead of `n` and demeaning about the subset mean instead of
about the target makes the bias a function of the hit rate, so it flips sign as drift
changes and silently reorders a strategy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12
#: Below this, annualisation is arithmetically valid and substantively meaningless.
MIN_YEARS_FOR_ANNUALISATION = 0.25


def daily_simple_rf(rf_continuous: pd.Series) -> pd.Series:
    """Convert an annualised continuously-compounded rate to a daily simple return."""
    return np.expm1(rf_continuous / TRADING_DAYS)


@dataclass
class Performance:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    elapsed_years: float

    cumulative_return: float
    cumulative_return_check: float
    cagr: float
    arithmetic_annual_return: float

    annual_vol: float
    downside_vol: float
    sharpe_daily: float
    sharpe_monthly: float
    sharpe_handout_convention: float
    sortino: float

    max_drawdown: float
    max_drawdown_date: pd.Timestamp
    best_day: float
    worst_day: float
    mean_rf: float
    calendar_years: pd.Series

    @property
    def cumulative_return_gap(self) -> float:
        """Float-stability residual on a telescoping sum. NOT independent corroboration.

        See the module docstring: the two expressions are algebraically identical, so this
        can only ever detect floating-point pathology, never a wrong equity curve.
        """
        return abs(self.cumulative_return - self.cumulative_return_check)

    @property
    def sharpe_gap(self) -> float:
        """Daily vs monthly Sharpe. This one IS a real check."""
        return abs(self.sharpe_daily - self.sharpe_monthly)

    def verify(self, tol: float = 1e-9) -> None:
        """Raise on floating-point pathology in the cumulative-return arithmetic.

        Deliberately narrow. This used to be presented as verifying the return itself,
        which it cannot do. Real verification of an equity curve is
        `BacktestResult.assert_identity`; real verification of the risk-adjusted numbers
        is the daily-vs-monthly Sharpe pair.
        """
        if self.cumulative_return_gap > tol:
            raise AssertionError(
                f"{self.label}: cumulative return arithmetic is numerically unstable "
                f"({self.cumulative_return:.6%} vs {self.cumulative_return_check:.6%}, "
                f"gap {self.cumulative_return_gap:.2e}). Note this is a float-stability "
                f"check only; it cannot detect a wrong input curve."
            )

    def render(self) -> str:
        return "\n".join(
            [
                f"{self.label}",
                f"  window                 {self.start:%Y-%m-%d} .. {self.end:%Y-%m-%d} "
                f"({self.years:.2f}y by bar count, {self.elapsed_years:.2f}y elapsed)",
                f"  cumulative return      {self.cumulative_return:>9.2%}",
                f"  CAGR                   {self.cagr:>9.2%}",
                f"  arithmetic annual      {self.arithmetic_annual_return:>9.2%}",
                f"  annualised volatility  {self.annual_vol:>9.2%}",
                f"  Sharpe (daily)         {self.sharpe_daily:>9.2f}",
                f"  Sharpe (monthly)       {self.sharpe_monthly:>9.2f}   "
                f"(gap {self.sharpe_gap:.2f})",
                f"  Sharpe (handout conv.) {self.sharpe_handout_convention:>9.2f}",
                f"  Sortino                {self.sortino:>9.2f}   "
                f"(target downside deviation, MAR = rf)",
                f"  max drawdown           {self.max_drawdown:>9.2%}   "
                f"on {self.max_drawdown_date:%Y-%m-%d}",
                f"  best / worst day       {self.best_day:>9.2%} / {self.worst_day:.2%}",
                f"  mean risk-free         {self.mean_rf:>9.2%}",
            ]
        )


def summarise(
    value: pd.Series,
    rf_continuous: pd.Series | None = None,
    label: str = "strategy",
) -> Performance:
    """Compute every statistic for an equity curve.

    `value` is a portfolio value series, not a return series. `rf_continuous` is the
    annualised continuously-compounded risk-free rate per bar; when omitted, excess
    returns equal raw returns and Sharpe becomes an information ratio against zero.
    """
    value = value.dropna()
    if len(value) < 3:
        raise ValueError(f"need at least 3 observations, got {len(value)}")
    if (value <= 0).any():
        raise ValueError("portfolio value must stay positive to compute returns")

    ret = value.pct_change().dropna()
    n = len(ret)
    years = n / TRADING_DAYS
    # Annualising a handful of bars produces numbers that are arithmetically correct and
    # completely meaningless: three observations rising 10% annualise to a CAGR of
    # 16,423,877%. Refuse rather than emit it.
    if years < MIN_YEARS_FOR_ANNUALISATION:
        raise ValueError(
            f"{label}: {n} return observations is {years:.4f} years, too short to "
            f"annualise (minimum {MIN_YEARS_FOR_ANNUALISATION}). CAGR and Sharpe would "
            f"be arithmetically valid and substantively meaningless."
        )

    # --- cumulative return, two independent ways
    cumulative = value.iloc[-1] / value.iloc[0] - 1.0
    cumulative_check = float(np.expm1(np.log1p(ret).sum()))

    cagr = (1.0 + cumulative) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    arithmetic_annual = (1.0 + ret.mean()) ** TRADING_DAYS - 1.0

    annual_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))

    if rf_continuous is None:
        rf_daily = pd.Series(0.0, index=ret.index)
        mean_rf = 0.0
    else:
        rf_daily = daily_simple_rf(rf_continuous).reindex(ret.index).ffill().fillna(0.0)
        mean_rf = float(rf_continuous.reindex(value.index).ffill().mean())

    excess = ret - rf_daily

    # --- Sharpe from daily sampling (the textbook definition)
    sd = float(excess.std(ddof=1))
    sharpe_daily = float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan

    # --- Sharpe from monthly sampling: different arithmetic, same quantity
    monthly_value = value.resample("ME").last().dropna()
    monthly_ret = monthly_value.pct_change().dropna()
    monthly_rf = (
        (1.0 + rf_daily).resample("ME").prod() - 1.0
    ).reindex(monthly_ret.index).fillna(0.0)
    monthly_excess = monthly_ret - monthly_rf
    sm = float(monthly_excess.std(ddof=1))
    sharpe_monthly = (
        float(monthly_excess.mean() / sm * np.sqrt(MONTHS))
        if sm > 0 and len(monthly_excess) > 2
        else np.nan
    )

    # --- the handout's convention, reported so the difference is visible
    sharpe_handout = (
        (arithmetic_annual - mean_rf) / annual_vol if annual_vol > 0 else np.nan
    )

    # Target downside deviation: deviations below the target (excess return of zero,
    # i.e. MAR = the risk-free rate), squared, averaged over ALL periods, not just the
    # losing ones. Using the negative subset's stdev instead makes the bias a function of
    # the hit rate: measured on synthetic paths it ranges from 1.49x (overstating, at a
    # 69% loss rate) to 0.61x (understating, at 17%), which silently reorders any
    # cross-strategy comparison.
    shortfall = excess.clip(upper=0.0)
    downside_vol = float(np.sqrt((shortfall**2).mean()) * np.sqrt(TRADING_DAYS))
    sortino = (
        float(excess.mean() * TRADING_DAYS / downside_vol)
        if downside_vol > 0
        else np.nan
    )

    # Elapsed calendar time, reported alongside the return-count basis. `years = n/252`
    # silently overstates CAGR for any series with missing sessions, and a reader seeing
    # "4.97y" next to a date range will assume calendar.
    elapsed_years = (value.index[-1] - value.index[0]).days / 365.25

    drawdown = value / value.cummax() - 1.0
    calendar = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
    calendar.index.name = "year"

    return Performance(
        label=label,
        start=value.index[0],
        end=value.index[-1],
        years=years,
        elapsed_years=elapsed_years,
        cumulative_return=float(cumulative),
        cumulative_return_check=cumulative_check,
        cagr=float(cagr),
        arithmetic_annual_return=float(arithmetic_annual),
        annual_vol=annual_vol,
        downside_vol=downside_vol,
        sharpe_daily=sharpe_daily,
        sharpe_monthly=sharpe_monthly,
        sharpe_handout_convention=float(sharpe_handout),
        sortino=sortino,
        max_drawdown=float(drawdown.min()),
        max_drawdown_date=drawdown.idxmin(),
        best_day=float(ret.max()),
        worst_day=float(ret.min()),
        mean_rf=mean_rf,
        calendar_years=calendar,
    )


def comparison_table(results: dict[str, Performance]) -> pd.DataFrame:
    """Side-by-side frame of the headline statistics."""
    rows = {
        name: {
            "cumulative return": p.cumulative_return,
            "CAGR": p.cagr,
            "annualised vol": p.annual_vol,
            "Sharpe (daily)": p.sharpe_daily,
            "Sharpe (monthly)": p.sharpe_monthly,
            "Sortino": p.sortino,
            "max drawdown": p.max_drawdown,
        }
        for name, p in results.items()
    }
    return pd.DataFrame(rows)
