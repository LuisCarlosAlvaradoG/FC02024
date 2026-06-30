"""
quantlib — Options Analytics Dashboard core library.

Two pricing engines (Black & Scholes, Heston), implied-vol inversion,
Heston calibration, market-data pipeline, book-level risk, and a per-contract
engine recommender. The Streamlit app in ../app.py is a thin presentation
layer over these modules.
"""
from .black_scholes import bs_price, bs_greeks, Greeks
from .heston import HestonParams, heston_price, heston_greeks
from .implied_vol import implied_vol
from .calibration import (calibrate_heston, CalibrationResult,
                          stability_across_days)
from .recommend import recommend_engine, Recommendation
from .portfolio import Position, aggregate_book, BookRisk
from . import data, theme

__all__ = [
    "bs_price", "bs_greeks", "Greeks",
    "HestonParams", "heston_price", "heston_greeks",
    "implied_vol",
    "calibrate_heston", "CalibrationResult", "stability_across_days",
    "recommend_engine", "Recommendation",
    "Position", "aggregate_book", "BookRisk",
    "data", "theme",
]
