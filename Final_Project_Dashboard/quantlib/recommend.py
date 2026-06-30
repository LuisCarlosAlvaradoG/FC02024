"""
Per-contract engine recommendation  (Feedback 1 - the derivatives trader).

A trader does not use one model for everything. Short-dated, at-the-money,
liquid contracts: B&S is fast and good enough. Long-dated or out-of-the-money
wings, where the skew dominates: trust the calibrated Heston.

This module returns a recommended engine plus a one-line, defensible
justification built from moneyness, maturity, and liquidity — the exact
artefact the brief asks the intern to defend live.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Recommendation:
    engine: str            # "Black-Scholes" or "Heston"
    confidence: str        # "High" / "Medium" / "Low"
    reason: str

    def as_dict(self):
        return {"engine": self.engine, "confidence": self.confidence,
                "reason": self.reason}


_RANK = {"High": 2, "Medium": 1, "Low": 0}


def _downgrade(conf, floor):
    """Lower confidence to at most `floor`."""
    return floor if _RANK[floor] < _RANK[conf] else conf


def recommend_engine(S, K, tau, *, option_type="call", moneyness_atm=0.05,
                     short_dated=30 / 365, long_dated=180 / 365,
                     wing=0.10, rel_spread=None, calibration_rmse=None):
    """
    Decide which engine to trust for a single contract.

    Heuristics (transparent on purpose, so it can be defended out loud):
    - near-ATM AND short maturity            -> B&S (flat vol is fine ATM).
    - intrinsic-dominated deep-ITM           -> B&S (negligible vega, skew irrelevant).
    - OTM-wing strike OR long maturity        -> Heston (skew / term structure).
    - else                                    -> Heston, with a skew-correction caveat.
    Liquidity and calibration fit then adjust confidence. The justification
    ALWAYS reports moneyness, maturity and liquidity, per the brief.
    """
    # Guard degenerate inputs (expiry day / bad quote).
    if not (S > 0 and K > 0) or tau <= 1e-6:
        return Recommendation("Black-Scholes", "Low",
                              "at/near expiry or invalid quote: trust intrinsic, "
                              "not a stochastic-vol model")

    signed_lm = np.log(K / S)               # >0: high strike, <0: low strike
    lm = abs(signed_lm)
    # Is the contract in-the-money? (calls ITM when S>K; puts ITM when S<K)
    itm = (signed_lm < 0) if option_type == "call" else (signed_lm > 0)
    deep = lm > 0.15
    near_atm = lm <= moneyness_atm
    is_short = tau <= short_dated
    is_long = tau >= long_dated
    is_wing = lm > wing

    # Maturity phrase (always reported).
    mat = (f"short-dated ({tau*365:.0f}d)" if is_short else
           f"long-dated ({tau*365:.0f}d)" if is_long else
           f"intermediate maturity ({tau*365:.0f}d)")
    # Moneyness phrase (always reported, ITM/OTM aware).
    side = "ATM" if near_atm else (
        f"{'deep ' if deep else ''}{'ITM' if itm else 'OTM'} ({K/S:.2f} K/S)")

    if near_atm and is_short:
        engine, conf = "Black-Scholes", "High"
        reasons = [f"{side}, {mat}: flat-vol B&S is accurate and fast"]
    elif deep and itm:
        engine, conf = "Black-Scholes", "High"
        reasons = [f"{side}, {mat}: intrinsic-dominated, negligible vega — "
                   "skew is irrelevant, B&S is fine"]
    elif is_wing or is_long:
        engine, conf = "Heston", "High"
        bits = []
        if is_wing:
            bits.append(f"{side} wing where the skew matters")
        if is_long:
            bits.append("term-structure of vol is in play")
        reasons = [f"{', '.join(bits)} ({mat})"]
    else:
        engine, conf = "Heston", "Medium"
        reasons = [f"{side}, {mat}: skew correction still helps"]

    # Liquidity is ALWAYS stated (requirement), and downgrades trust if wide.
    if rel_spread is None or not np.isfinite(rel_spread):
        reasons.append("liquidity unknown (no spread)")
        conf = _downgrade(conf, "Medium")
    elif rel_spread > 0.20:
        reasons.append(f"wide market ({rel_spread:.0%} spread) — quote is noisy")
        conf = _downgrade(conf, "Low" if rel_spread > 0.50 else "Medium")
    else:
        reasons.append(f"liquid ({rel_spread:.0%} spread)")

    # Calibration fit caveat only matters when we lean on Heston.
    if engine == "Heston" and calibration_rmse is not None and \
            np.isfinite(calibration_rmse) and calibration_rmse > 0.03:
        reasons.append(f"calibration RMSE {calibration_rmse:.2%} vol — fit is loose")
        conf = _downgrade(conf, "Medium")

    return Recommendation(engine, conf, "; ".join(reasons))
