"""Cox-Ross-Rubinstein binomial pricer: an independent second opinion on bs.py.

This exists to be disagreed with. A closed-form implementation and a lattice
implementation share no arithmetic, so when they agree to 1e-3 on European options
the closed form is almost certainly transcribed correctly. That is the same
multi-engine-agreement discipline applied elsewhere in this project, at the level
of a single pricing formula.

The American flag is not just for tests. A covered call written on a dividend payer
faces genuine early assignment just before the ex-dividend date, and the gap between
the American and European price is the size of that ignored risk.
"""

from __future__ import annotations

import math

from .bs import Kind, intrinsic


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: Kind,
    q: float = 0.0,
    steps: int = 2000,
    american: bool = False,
) -> float:
    """Binomial option price. Converges to Black-Scholes as `steps` grows."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    if T <= 0 or sigma <= 0:
        return intrinsic(S, K, kind)

    dt = T / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    # Risk-neutral probability under a continuous dividend yield.
    p = (math.exp((r - q) * dt) - down) / (up - down)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside [0,1]: "
            "sigma is too small for this drift and step count"
        )
    discount = math.exp(-r * dt)

    # Terminal layer, then fold backwards in place.
    values = [
        intrinsic(S * (up**j) * (down ** (steps - j)), K, kind)
        for j in range(steps + 1)
    ]

    for step in range(steps - 1, -1, -1):
        for j in range(step + 1):
            values[j] = discount * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                spot = S * (up**j) * (down ** (step - j))
                values[j] = max(values[j], intrinsic(spot, K, kind))

    return values[0]
