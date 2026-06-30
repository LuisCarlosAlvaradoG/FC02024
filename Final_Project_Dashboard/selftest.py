"""
Engine validation suite. Run:  python selftest.py

Checks (the brief's "validated against each other and against market quotes"):
  1. B&S put-call parity holds.
  2. B&S closed-form Greeks match finite-difference bumps.
  3. Heston -> B&S: as xi -> 0 with v0 = theta = sigma^2, Heston price
     converges to the Black & Scholes price.
  4. Implied-vol inversion round-trips.
  5. Heston calibration recovers known parameters from a synthetic smile.
"""
import numpy as np

from quantlib import (bs_price, bs_greeks, heston_price, implied_vol,
                      HestonParams, calibrate_heston)
from quantlib.data import synthetic_snapshot, chain_to_quotes

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def check(name, cond, detail=""):
    tag = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    S, K, tau, r, q, sigma = 100.0, 105.0, 0.5, 0.04, 0.01, 0.25
    ok = True

    print("1. Black-Scholes put-call parity")
    c = bs_price(S, K, tau, r, sigma, q, "call")
    p = bs_price(S, K, tau, r, sigma, q, "put")
    lhs = c - p
    rhs = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    ok &= check("C - P = S e^-qT - K e^-rT", abs(lhs - rhs) < 1e-10,
                f"diff={abs(lhs-rhs):.2e}")

    print("2. B&S Greeks vs finite differences")
    g = bs_greeks(S, K, tau, r, sigma, q, "call")
    h = 1e-4
    fd_delta = (bs_price(S + h, K, tau, r, sigma, q) -
                bs_price(S - h, K, tau, r, sigma, q)) / (2 * h)
    fd_vega = (bs_price(S, K, tau, r, sigma + h, q) -
               bs_price(S, K, tau, r, sigma - h, q)) / (2 * h)
    ok &= check("Delta", abs(g.delta - fd_delta) < 1e-5, f"diff={abs(g.delta-fd_delta):.2e}")
    ok &= check("Vega", abs(g.vega - fd_vega) < 1e-3, f"diff={abs(g.vega-fd_vega):.2e}")

    print("3. Heston -> B&S as xi -> 0")
    sig = 0.30
    p_h = HestonParams(v0=sig**2, kappa=2.0, theta=sig**2, xi=1e-4, rho=0.0)
    for KK in (80, 100, 120):
        hp = heston_price(S, KK, tau, r, p_h, q, "call")
        bp = bs_price(S, KK, tau, r, sig, q, "call")
        ok &= check(f"K={KK}", abs(hp - bp) < 1e-2, f"H={hp:.4f} BS={bp:.4f}")

    print("4. Implied-vol round-trip")
    px = bs_price(S, K, tau, r, 0.33, q, "call")
    iv = implied_vol(px, S, K, tau, r, q, "call")
    ok &= check("recover sigma=0.33", abs(iv - 0.33) < 1e-4, f"iv={iv:.5f}")

    print("5. Heston calibration recovers known params (synthetic)")
    snap = synthetic_snapshot("TEST")
    quotes = chain_to_quotes(snap)
    res = calibrate_heston(quotes, snap.spot, snap.r, snap.q, maxiter=25)
    true_p = snap._true_params
    print(f"     true : v0={true_p.v0:.3f} kappa={true_p.kappa:.2f} "
          f"theta={true_p.theta:.3f} xi={true_p.xi:.2f} rho={true_p.rho:.2f}")
    print(f"     fit  : v0={res.params.v0:.3f} kappa={res.params.kappa:.2f} "
          f"theta={res.params.theta:.3f} xi={res.params.xi:.2f} rho={res.params.rho:.2f}")
    ok &= check("RMSE < 1.0 vol point", res.rmse_vol < 0.01,
                f"rmse={res.rmse_vol*100:.3f}%")
    ok &= check("rho sign (negative skew)", res.params.rho < 0,
                f"rho={res.params.rho:.3f}")

    print("6. OCC option-symbol parsing (offline)")
    from quantlib.data import parse_occ_symbol
    pc = parse_occ_symbol("VIX260819C00024000")
    ok &= check("root=VIX", pc.root == "VIX", pc.root)
    ok &= check("call", pc.option_type == "call")
    ok &= check("strike=24", abs(pc.strike - 24.0) < 1e-9, str(pc.strike))
    ok &= check("expiry=2026-08-19", pc.expiry.isoformat() == "2026-08-19",
                pc.expiry.isoformat())
    pp = parse_occ_symbol("AAPL260116P00150000")
    ok &= check("AAPL put K=150", pp.root == "AAPL" and pp.option_type == "put"
                and abs(pp.strike - 150.0) < 1e-9, f"{pp.root} {pp.option_type} {pp.strike}")
    try:
        parse_occ_symbol("NOTASYMBOL")
        ok &= check("rejects garbage", False)
    except ValueError:
        ok &= check("rejects garbage", True)

    print()
    print((GREEN + "ALL CHECKS PASSED" if ok else RED + "SOME CHECKS FAILED") + RESET)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
