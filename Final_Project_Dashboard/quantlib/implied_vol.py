"""
Black & Scholes implied-volatility inversion.

Uses Brent's method on the price residual. Returns NaN for arbitrage-violating
or non-invertible quotes (e.g. price below intrinsic) so dirty market rows can
be filtered downstream rather than poisoning the smile.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import bs_price


def implied_vol(price, S, K, tau, r, q=0.0, option_type="call",
                lo=1e-4, hi=5.0):
    """Invert the B&S price for sigma. NaN when no solution in [lo, hi]."""
    if price is None or not np.isfinite(price) or price <= 0 or tau <= 0:
        return np.nan

    # Intrinsic / no-arbitrage bounds.
    df_r, df_q = np.exp(-r * tau), np.exp(-q * tau)
    if option_type == "call":
        intrinsic = max(S * df_q - K * df_r, 0.0)
        upper = S * df_q
    else:
        intrinsic = max(K * df_r - S * df_q, 0.0)
        upper = K * df_r
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan

    def f(sigma):
        return bs_price(S, K, tau, r, sigma, q, option_type) - price

    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan
