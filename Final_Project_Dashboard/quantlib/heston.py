"""
Heston (1993) stochastic-volatility pricing engine.

Implementation uses the *little Heston trap* characteristic function
(Albrecher, Mayer, Schoutens & Tistaert, 2007) which selects the stable
branch of the complex square root and keeps the price continuous in the
integration variable. This is the formulation referenced in the brief.

Model
-----
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Parameters (calibrated, not observed): v0, kappa, theta, xi, rho.

Pricing follows the two-probability decomposition
    C = S e^{-q tau} P1 - K e^{-r tau} P2
with P_j recovered by Gil-Pelaez inversion of the characteristic function.
Greeks are obtained by central finite differences (bump-and-reprice); the
bump method is documented so it can be made consistent with the B&S engine
when the desk wants an apples-to-apples comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .black_scholes import Greeks

# Fixed Gauss-Legendre quadrature on [0, U_MAX]. A 128-point rule on a 200-wide
# truncation prices vanilla Heston to ~1e-6 and, unlike adaptive scipy.quad,
# evaluates the characteristic function vectorially — fast enough to recalibrate
# live. Nodes/weights are built once at import.
_U_MAX = 200.0
_GL_N = 128
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_U_NODES = 0.5 * _U_MAX * (_gl_x + 1.0)            # map [-1,1] -> [0, U_MAX]
_U_WEIGHTS = 0.5 * _U_MAX * _gl_w


@dataclass
class HestonParams:
    v0: float       # initial variance
    kappa: float    # mean-reversion speed
    theta: float    # long-run variance
    xi: float       # vol of vol
    rho: float      # spot/vol correlation

    def feller(self) -> float:
        """Feller quantity 2*kappa*theta - xi^2. Feller holds when >= 0."""
        return 2.0 * self.kappa * self.theta - self.xi ** 2

    def feller_ok(self) -> bool:
        return self.feller() >= 0.0

    def as_dict(self):
        return asdict(self)


def _cf_trap(u, tau, r, q, p: HestonParams, j: int):
    """Heston characteristic function (little-trap form) for probability j."""
    if j == 1:
        b = p.kappa - p.rho * p.xi
        uj = 0.5
    else:
        b = p.kappa
        uj = -0.5

    rho_xi_iu = p.rho * p.xi * 1j * u
    d = np.sqrt((rho_xi_iu - b) ** 2 - p.xi ** 2 * (2.0 * uj * 1j * u - u ** 2))
    # little-trap: g = (b - rho*xi*i*u - d) / (b - rho*xi*i*u + d)
    num = b - rho_xi_iu - d
    den = b - rho_xi_iu + d
    g = num / den
    exp_dt = np.exp(-d * tau)

    C = (r - q) * 1j * u * tau + (p.kappa * p.theta / p.xi ** 2) * (
        num * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = (num / p.xi ** 2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(C + D * p.v0)


def _cf_terms(tau, r, q, p: HestonParams, j: int, x):
    """Vectorised exp(C + D v0 + i u x) evaluated at all quadrature nodes."""
    return _cf_trap(_U_NODES, tau, r, q, p, j) * np.exp(1j * _U_NODES * x)


def heston_call_prices(S, strikes, tau, r, p: HestonParams, q=0.0):
    """
    Vectorised Heston *call* prices for an array of strikes at one maturity.

    The characteristic functions are independent of K, so they are evaluated
    once per maturity and reused across all strikes — this is what makes
    calibration tractable.
    """
    tau = max(float(tau), 1e-10)
    strikes = np.atleast_1d(np.asarray(strikes, float))
    x = np.log(S)
    cf1 = _cf_terms(tau, r, q, p, 1, x)          # (N_nodes,)
    cf2 = _cf_terms(tau, r, q, p, 2, x)
    lnK = np.log(strikes)[:, None]               # (N_K, 1)
    u = _U_NODES[None, :]
    w = _U_WEIGHTS[None, :]
    int1 = (np.exp(-1j * u * lnK) * cf1[None, :] / (1j * u)).real
    int2 = (np.exp(-1j * u * lnK) * cf2[None, :] / (1j * u)).real
    P1 = 0.5 + (int1 * w).sum(axis=1) / np.pi
    P2 = 0.5 + (int2 * w).sum(axis=1) / np.pi
    call = S * np.exp(-q * tau) * P1 - strikes * np.exp(-r * tau) * P2
    return np.maximum(call, 0.0)


def heston_price(S, K, tau, r, p: HestonParams, q=0.0, option_type="call"):
    """European option price under Heston (put via put-call parity)."""
    tau = max(float(tau), 1e-10)
    call = float(heston_call_prices(S, K, tau, r, p, q)[0])
    if option_type == "call":
        return call
    # put-call parity: P = C - S e^{-q tau} + K e^{-r tau}
    return call - S * np.exp(-q * tau) + K * np.exp(-r * tau)


def heston_greeks(S, K, tau, r, p: HestonParams, q=0.0, option_type="call",
                  ds=None, dvol=None, dr=1e-4, dt=1.0 / 365.0) -> Greeks:
    """
    Bump-and-reprice Greeks. Vol bumps act on v0 mapped through an equivalent
    spot-vol level sqrt(v0) so that Vega/Vanna/Volga are reported per unit of
    volatility (same convention as the B&S engine).
    """
    ds = ds if ds is not None else max(S * 1e-4, 1e-4)
    sig0 = np.sqrt(max(p.v0, 1e-12))
    dvol = dvol if dvol is not None else 1e-3

    def price_at(Sx=S, params=p, rx=r, taux=tau):
        return heston_price(Sx, K, taux, rx, params, q, option_type)

    base = price_at()

    # Delta / Gamma (spot bumps)
    up, dn = price_at(Sx=S + ds), price_at(Sx=S - ds)
    delta = (up - dn) / (2 * ds)
    gamma = (up - 2 * base + dn) / (ds ** 2)

    # Vega / Volga (bump implied spot-vol level, re-map to v0)
    p_up = HestonParams(**{**p.as_dict(), "v0": (sig0 + dvol) ** 2})
    p_dn = HestonParams(**{**p.as_dict(), "v0": max(sig0 - dvol, 1e-6) ** 2})
    v_up, v_dn = price_at(params=p_up), price_at(params=p_dn)
    vega = (v_up - v_dn) / (2 * dvol)
    volga = (v_up - 2 * base + v_dn) / (dvol ** 2)

    # Vanna: cross derivative d^2 Price / dS dvol
    p_up_s_up = heston_price(S + ds, K, tau, r, p_up, q, option_type)
    p_up_s_dn = heston_price(S - ds, K, tau, r, p_up, q, option_type)
    p_dn_s_up = heston_price(S + ds, K, tau, r, p_dn, q, option_type)
    p_dn_s_dn = heston_price(S - ds, K, tau, r, p_dn, q, option_type)
    vanna = (p_up_s_up - p_up_s_dn - p_dn_s_up + p_dn_s_dn) / (4 * ds * dvol)

    # Rho (rate bump)
    rho = (price_at(rx=r + dr) - price_at(rx=r - dr)) / (2 * dr)

    # Theta (calendar decay): tau decreases as time passes
    tau_dn = max(tau - dt, 1e-6)
    theta = (price_at(taux=tau_dn) - base) / dt

    return Greeks(base, delta, gamma, vega, theta, rho, vanna, volga)
