"""
Book-level risk aggregation  (Feedback 2 - the risk manager).

"Greeks without a portfolio are an academic toy." Risk lives at the book.
This module nets Greeks across a small portfolio of options and answers the
two questions the risk manager demanded:

    1. What is the delta hedge (shares of the underlying to neutralise delta)?
    2. How much does it cost to stay gamma-neutral (using a liquid hedging
       option), and what residual delta does that leave?

Greeks are computed via a single, explicitly chosen method per engine so the
book is *internally consistent* — the risk manager's red flag. The caller
passes a `greek_fn(position) -> Greeks`; the dashboard wires this to either the
closed-form B&S Greeks or the bumped Heston Greeks, but never mixes them
silently within one book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .black_scholes import Greeks


@dataclass
class Position:
    option_type: str       # "call" / "put"
    K: float
    tau: float
    qty: float             # signed; +long / -short, in contracts
    multiplier: float = 100.0
    label: str = ""


@dataclass
class BookRisk:
    net: dict                       # netted Greeks (display units)
    per_position: list              # list of (Position, Greeks display dict)
    spot: float
    delta_hedge_shares: float       # shares of underlying to flatten delta
    gamma_neutral: dict = field(default_factory=dict)


def aggregate_book(positions, spot, greek_fn: Callable[[Position], Greeks],
                   hedge_option: Position = None,
                   hedge_greek_fn: Callable[[Position], Greeks] = None):
    """
    Net the book's Greeks and compute hedges.

    `greek_fn` must use ONE consistent method for every position (do not mix
    closed-form and bumped Greeks inside a single book).
    """
    keys = ["Delta", "Gamma", "Vega (1%)", "Theta (1d)", "Rho (1%)",
            "Vanna", "Volga"]
    net = {k: 0.0 for k in keys}
    net["Price"] = 0.0
    per_position = []

    for pos in positions:
        g = greek_fn(pos).display()
        scaled = {k: g[k] * pos.qty * pos.multiplier for k in g}
        for k in net:
            net[k] += scaled.get(k, 0.0)
        per_position.append((pos, g))

    # Delta hedge: short net delta in the underlying (1 share per delta unit).
    delta_hedge_shares = -net["Delta"]

    result = BookRisk(
        net=net, per_position=per_position, spot=spot,
        delta_hedge_shares=delta_hedge_shares,
    )

    # Gamma-neutral overlay using a liquid hedging option.
    if hedge_option is not None:
        hf = hedge_greek_fn or greek_fn
        hg = hf(hedge_option).display()
        unit_gamma = hg["Gamma"] * hedge_option.multiplier
        unit_delta = hg["Delta"] * hedge_option.multiplier
        if abs(unit_gamma) > 1e-12:
            n_hedge = -net["Gamma"] / unit_gamma          # contracts of hedge option
            residual_delta = net["Delta"] + n_hedge * unit_delta
            cost = n_hedge * hg["Price"] * hedge_option.multiplier
            result.gamma_neutral = {
                "hedge_contracts": n_hedge,
                "residual_delta": residual_delta,
                "residual_delta_hedge_shares": -residual_delta,
                "cost": cost,
                "hedge_label": hedge_option.label or
                f"{hedge_option.option_type} K={hedge_option.K}",
            }
    return result
