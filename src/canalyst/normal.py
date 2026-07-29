"""Standard normal CDF and inverse CDF.

This is the seam that keeps scipy out of the dependency list. `statistics.NormalDist`
has been in the stdlib since 3.8 and covers everything Black-Scholes needs:
`norm.cdf` -> `cdf`, `norm.ppf` -> `inv_cdf`. Both are scalar, which is fine because
pricing happens once per option per bar, not in a hot loop.

Keeping it in one module means a future need for vectorised erf has exactly one
place to change.
"""

from __future__ import annotations

from statistics import NormalDist

_STANDARD = NormalDist(0.0, 1.0)


def cdf(x: float) -> float:
    """P(Z <= x) for standard normal Z."""
    return _STANDARD.cdf(x)


def inv_cdf(p: float) -> float:
    """Quantile function. Requires 0 < p < 1; raises otherwise, deliberately.

    A silent clamp here would turn a nonsensical delta target into a plausible
    strike, which is the failure mode this project exists to avoid.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"inv_cdf requires 0 < p < 1, got {p!r}")
    return _STANDARD.inv_cdf(p)
