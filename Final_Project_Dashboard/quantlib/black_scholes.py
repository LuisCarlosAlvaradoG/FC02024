"""
Black & Scholes pricing engine with closed-form Greeks (incl. continuous
dividend yield q, plus second-order Vanna and Volga).

Conventions
-----------
- Continuous compounding. tau is time to maturity in years.
- sigma, r, q are annualised, expressed as decimals (0.20 == 20%).
- Greeks are returned in *natural* (per-unit) terms. The display layer is
  responsible for scaling (Vega/Rho per 1% move, Theta per calendar day).

The closed forms here are the benchmark every other engine is validated
against, exactly as the mandate requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_SQRT_EPS = 1e-12


def _d1_d2(S, K, tau, r, q, sigma):
    """Return (d1, d2) with a numerical floor on tau and sigma."""
    tau = np.maximum(tau, _SQRT_EPS)
    sigma = np.maximum(sigma, _SQRT_EPS)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / vol
    d2 = d1 - vol
    return d1, d2


def bs_price(S, K, tau, r, sigma, q=0.0, option_type="call"):
    """Black & Scholes price for a European call or put."""
    d1, d2 = _d1_d2(S, K, tau, r, q, sigma)
    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    if option_type == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif option_type == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


@dataclass
class Greeks:
    """First- and second-order sensitivities in natural units."""
    price: float
    delta: float
    gamma: float
    vega: float        # per 1.00 change in sigma  (d Price / d sigma)
    theta: float       # per year (negative for long options, typically)
    rho: float         # per 1.00 change in r
    vanna: float       # d Vega / d Spot  == d Delta / d sigma
    volga: float       # d Vega / d sigma (a.k.a. Vomma)

    def display(self):
        """Greeks scaled to desk conventions for tables/plots."""
        return {
            "Price": self.price,
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Vega (1%)": self.vega / 100.0,
            "Theta (1d)": self.theta / 365.0,
            "Rho (1%)": self.rho / 100.0,
            "Vanna": self.vanna / 100.0,
            "Volga": self.volga / 100.0,
        }


def bs_greeks(S, K, tau, r, sigma, q=0.0, option_type="call") -> Greeks:
    """Closed-form Greeks for a European option under Black & Scholes."""
    tau = max(float(tau), _SQRT_EPS)
    sigma = max(float(sigma), _SQRT_EPS)
    d1, d2 = _d1_d2(S, K, tau, r, q, sigma)
    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    pdf_d1 = norm.pdf(d1)
    sqrt_tau = np.sqrt(tau)

    price = bs_price(S, K, tau, r, sigma, q, option_type)
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_tau)
    vega = S * df_q * pdf_d1 * sqrt_tau
    vanna = -df_q * pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    if option_type == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            -S * df_q * pdf_d1 * sigma / (2 * sqrt_tau)
            - r * K * df_r * norm.cdf(d2)
            + q * S * df_q * norm.cdf(d1)
        )
        rho = K * tau * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (
            -S * df_q * pdf_d1 * sigma / (2 * sqrt_tau)
            + r * K * df_r * norm.cdf(-d2)
            - q * S * df_q * norm.cdf(-d1)
        )
        rho = -K * tau * df_r * norm.cdf(-d2)

    return Greeks(price, delta, gamma, vega, theta, rho, vanna, volga)
