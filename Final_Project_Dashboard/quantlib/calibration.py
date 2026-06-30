"""
Heston calibration to a market implied-volatility smile/surface.

Per the quant developer's feedback, calibration is treated as the ill-posed
problem it is:

- Bounded global search (differential evolution) for a robust starting point,
  followed by a local least-squares polish (Trust Region Reflective).
- Sensible bounds and a Feller-aware penalty so the optimiser is nudged
  toward economically plausible parameters instead of garbage that happens to
  fit this one snapshot.
- The objective fits in *implied-vol* space (vega-comparable), which is far
  better conditioned than fitting raw prices.

The result carries diagnostics (RMSE, Feller quantity, success flag) so the
dashboard can be honest about fit quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .black_scholes import bs_price, bs_greeks
from .heston import HestonParams, heston_call_prices
from .implied_vol import implied_vol

# (v0, kappa, theta, xi, rho) bounds — wide but economically sane.
DEFAULT_BOUNDS = [
    (1e-3, 1.0),     # v0
    (1e-2, 10.0),    # kappa
    (1e-3, 1.0),     # theta
    (1e-2, 5.0),     # xi
    (-0.999, 0.0),   # rho  (equity skew => negative)
]


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_vol: float          # RMSE in implied-vol points
    n_quotes: int
    feller: float
    feller_ok: bool
    success: bool
    message: str
    pinned: tuple = ()       # parameter names sitting on a bound (degenerate fit)


def _market_to_arrays(quotes):
    """quotes: iterable of dicts with K, tau, iv, option_type, (weight)."""
    K = np.array([q["K"] for q in quotes], float)
    tau = np.array([q["tau"] for q in quotes], float)
    iv = np.array([q["iv"] for q in quotes], float)
    otype = [q.get("option_type", "call") for q in quotes]
    w = np.array([q.get("weight", 1.0) for q in quotes], float)
    return K, tau, iv, otype, w


def _model_prices(theta_vec, S, r, q, K, tau, otype):
    """Model option prices, pricing all strikes of each maturity at once."""
    p = HestonParams(*theta_vec)
    out = np.full(len(K), np.nan)
    for t in np.unique(tau):
        idx = np.where(tau == t)[0]
        calls = heston_call_prices(S, K[idx], t, r, p, q)
        for j, i in enumerate(idx):
            call = calls[j]
            if otype[i] == "put":          # put-call parity
                out[i] = call - S * np.exp(-q * t) + K[i] * np.exp(-r * t)
            else:
                out[i] = call
    return out


def _model_ivs(theta_vec, S, r, q, K, tau, otype):
    """Final-fit diagnostic: invert model prices to implied vol (one pass)."""
    prices = _model_prices(theta_vec, S, r, q, K, tau, otype)
    return np.array([implied_vol(prices[i], S, K[i], tau[i], r, q, otype[i])
                     for i in range(len(K))])


def calibrate_heston(quotes, S, r, q=0.0, bounds=None, feller_penalty=0.5,
                     maxiter=40, seed=42, polish=True):
    """
    Fit Heston parameters to market IVs.

    Returns a CalibrationResult. `feller_penalty` adds a soft penalty (in vol
    points) per unit of Feller violation; set to 0 to disable.
    """
    bounds = bounds or DEFAULT_BOUNDS
    K, tau, mkt_iv, otype, w = _market_to_arrays(quotes)
    valid = np.isfinite(mkt_iv)
    K, tau, mkt_iv, w = K[valid], tau[valid], mkt_iv[valid], w[valid]
    otype = [o for o, v in zip(otype, valid) if v]
    if len(K) < 5:
        raise ValueError("Need at least 5 clean quotes to calibrate Heston.")

    sqrt_w = np.sqrt(w)

    # Vega-weighted PRICE objective: residual_i = (model_px - mkt_px) / vega_i.
    # Dividing the price error by Black-Scholes vega converts it into an
    # (approximate) implied-vol error, so we fit in the well-conditioned vol
    # metric WITHOUT a per-strike root-find inside the optimiser loop. Market
    # prices and vegas come from the quoted IVs and are precomputed once.
    mkt_px = np.array([bs_price(S, K[i], tau[i], r, mkt_iv[i], q, otype[i])
                       for i in range(len(K))])
    mkt_vega = np.array([max(bs_greeks(S, K[i], tau[i], r, mkt_iv[i], q,
                                       otype[i]).vega, 1e-6)
                         for i in range(len(K))])

    def residuals(theta_vec):
        model_px = _model_prices(theta_vec, S, r, q, K, tau, otype)
        res = (model_px - mkt_px) / mkt_vega          # ~ implied-vol error
        res = np.where(np.isfinite(res), res, 1.0)    # penalise pricing failures
        res = res * sqrt_w
        if feller_penalty > 0:
            p = HestonParams(*theta_vec)
            viol = max(0.0, -p.feller())
            res = np.append(res, feller_penalty * viol)
        return res

    def cost(theta_vec):
        r_ = residuals(theta_vec)
        return float(np.sum(r_ ** 2))

    de = differential_evolution(
        cost, bounds, maxiter=maxiter, tol=1e-7, seed=seed,
        polish=False, updating="deferred", workers=1,
    )
    best = de.x
    success, message = de.success, de.message

    if polish:
        try:
            lsq = least_squares(
                residuals, best,
                bounds=(np.array([b[0] for b in bounds]),
                        np.array([b[1] for b in bounds])),
                xtol=1e-10, ftol=1e-10, max_nfev=200,
            )
            best = lsq.x
            # least_squares.status > 0 means a real convergence criterion was
            # met (xtol/ftol/gtol). Only status 0 (hit max_nfev) is a failure;
            # do NOT AND the local solver's strict `success` flag in, or a good
            # fit that simply used its eval budget gets a false "CHECK".
            success = success and (lsq.status > 0)
            message = f"{message} | polish: {lsq.message}"
        except Exception as exc:  # pragma: no cover - defensive
            message = f"{message} | polish failed: {exc}"

    p = HestonParams(*best)
    model_iv = _model_ivs(best, S, r, q, K, tau, otype)
    err = (model_iv - mkt_iv)
    err = err[np.isfinite(err)]
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan

    # Flag parameters pinned to a bound — a classic ill-posed/degenerate fit.
    names = ("v0", "kappa", "theta", "xi", "rho")
    pinned = tuple(
        nm for nm, val, (lo, hi) in zip(names, best, bounds)
        if abs(val - lo) <= 1e-3 * (hi - lo) or abs(val - hi) <= 1e-3 * (hi - lo))

    return CalibrationResult(
        params=p, rmse_vol=rmse, n_quotes=len(K),
        feller=p.feller(), feller_ok=p.feller_ok(),
        success=bool(success), message=str(message), pinned=pinned,
    )


def stability_across_days(quotes, S, r, q=0.0, n_days=4, iv_noise=0.005,
                          spot_noise=0.01, seed=11, maxiter=20):
    """
    Emulate the brief's "re-calibrate on two different days and compare".

    With a single market snapshot we cannot literally pull two trading days, so
    we generate plausible *re-quotes* for successive days by perturbing each
    implied vol by Gaussian noise (`iv_noise`, in vol points) and nudging the
    spot (`spot_noise`, relative). Re-calibrating on each perturbed day reveals
    how much the Heston parameters move for an economically tiny change in the
    market — the day-to-day instability the quant developer warned about.

    Returns: list of dicts (one per day) with params, rmse, feller, pinned, and
    a final 'spread' row summarising max-min dispersion per parameter.
    """
    rng = np.random.default_rng(seed)
    base = _market_to_arrays(quotes)
    names = ("v0", "kappa", "theta", "xi", "rho")
    rows = []
    for d in range(n_days):
        if d == 0:
            day_quotes, day_S = quotes, S        # day 0 = the actual market
        else:
            day_S = S * (1.0 + rng.normal(0, spot_noise))
            day_quotes = []
            for qd in quotes:
                qd2 = dict(qd)
                qd2["iv"] = max(qd["iv"] + rng.normal(0, iv_noise), 1e-3)
                day_quotes.append(qd2)
        try:
            res = calibrate_heston(day_quotes, day_S, r, q, maxiter=maxiter)
            p = res.params
            rows.append({"day": "today" if d == 0 else f"+{d}d",
                         "v0": p.v0, "kappa": p.kappa, "theta": p.theta,
                         "xi": p.xi, "rho": p.rho, "rmse": res.rmse_vol,
                         "feller_ok": res.feller_ok, "pinned": res.pinned})
        except Exception:
            rows.append({"day": f"+{d}d", "v0": np.nan, "kappa": np.nan,
                         "theta": np.nan, "xi": np.nan, "rho": np.nan,
                         "rmse": np.nan, "feller_ok": False, "pinned": ()})

    # Dispersion summary: max-min per parameter across days.
    spread = {"day": "range"}
    for nm in names:
        vals = np.array([row[nm] for row in rows], float)
        vals = vals[np.isfinite(vals)]
        spread[nm] = (vals.max() - vals.min()) if len(vals) else np.nan
    spread["rmse"] = np.nan
    spread["feller_ok"] = all(row["feller_ok"] for row in rows)
    spread["pinned"] = ()
    rows.append(spread)
    return rows
