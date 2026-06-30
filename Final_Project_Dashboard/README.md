# Options Analytics Terminal — B&S vs Heston

A Bloomberg-style options desk dashboard built for the Quantitative Finance
final project. Two pricing engines, real/offline market data, the volatility
smile as the centrepiece, full Greeks, book-level risk, a per-contract engine
recommendation, and honest Heston calibration with stability diagnostics.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Validate the engines first (recommended before the live defense):

```bash
python selftest.py     # numerical validation of both engines + calibration
python apptest.py      # headless smoke test: renders the whole dashboard
```

## Input modes (sidebar → MODE)

- **Underlying chain** — pick a ticker, then browse expiry / strike / call-put
  from the cleaned chain.
- **Specific contract** — paste a Yahoo/OCC option symbol
  (e.g. `VIX260819C00024000`). The app **parses** type / strike / expiry from
  the symbol (no redundant selectors), pulls that contract's **own real price
  history** from Yahoo, resolves the underlying spot, and derives IV (Yahoo's if
  present, otherwise inverted from the market price). You can **download the
  contract history as CSV**, and toggle **"Override parameters manually"** to
  hand-edit spot / strike / days / type / IV when the quote is dirty or missing.
  Dirty-quote handling is explicit (e.g. zero/crossed bid-ask falls back to the
  last traded price, and the substitution is flagged).

OCC symbol format: `ROOT + YYMMDD + C/P + strike×1000` (8 digits). Examples:
`AAPL260116C00150000` (AAPL call, 150 strike, 2026-01-16), `VIX260819C00024000`
(VIX call, 24 strike, 2026-08-19).

## Data modes (sidebar → DATA SOURCE)

- **synthetic** — fully offline. Builds an arbitrage-consistent smile from a
  known Heston parameter set, so every panel works with no network. Use this
  during the defense if Wi-Fi is unreliable.
- **live** — pulls a real option chain + spot from `yfinance`, the risk-free
  proxy from FRED (`DTB3`), cleans it, prices it, and **saves a snapshot** to
  `data/` for offline replay.
- **snapshot** — replays the last saved pull.

The pipeline degrades gracefully: live → snapshot → synthetic.

## Layout (`quantlib/`)

| Module             | Responsibility |
|--------------------|----------------|
| `black_scholes.py` | Closed-form pricing + Greeks (incl. Vanna, Volga, dividend yield). The benchmark. |
| `heston.py`        | Heston pricing via the *little Heston trap* characteristic function; bumped Greeks. |
| `implied_vol.py`   | Brent inversion with no-arbitrage guards. |
| `calibration.py`   | Differential evolution + least-squares polish, Feller-aware, fits in IV space. |
| `data.py`          | yfinance/FRED retrieval, cleaning, snapshotting, synthetic fallback. |
| `portfolio.py`     | Book-level net Greeks, delta hedge, gamma-neutral cost. |
| `recommend.py`     | Per-contract recommended engine + justification. |
| `theme.py`         | Bloomberg-terminal Plotly template + Streamlit CSS. |
| `app.py`           | Thin Streamlit presentation layer over the library. |

## How the brief maps to the code

**Minimum deliverable (scores to 100)**

- Panel 1 — Pricing & validation: `app.py` tab 1; both engines vs market mid,
  abs/rel error, call/put + strike/expiry selectors.
- Panel 2 — Greeks: tab 3; all Greeks vs spot, B&S closed-form vs Heston,
  second-order Vanna/Volga.
- Panel 3 — Volatility smile/surface: tab 2; market IV vs log-moneyness, the
  B&S flat line, the calibrated Heston smile, optional 3D surface.
- Data pipeline: `data.py` (retrieve + clean + snapshot, reproducible).
- Engines validated against each other and the market: `selftest.py`.

**Quant Team Feedback (extra credit)**

- **Feedback 1 (+3) — engine recommendation.** `recommend.py`; shown per
  contract in tab 1 with a one-line, defensible justification (moneyness,
  maturity, liquidity, calibration fit).
- **Feedback 2 (+3) — portfolio Greeks.** `portfolio.py`; tab 4 nets a 2–4 leg
  book, gives the delta hedge and the gamma-neutral cost. Greeks are computed
  with **one consistent engine per book** (the risk manager's red flag).
- **Feedback 3 (+3) — honest calibration.** `calibration.py`; differential
  evolution + bounded least squares (Feller-aware soft penalty), the Feller
  condition and bound-pinned parameters flagged in the UI, and **two stability
  checks** (tab 5): *across expiries* (term-structure) and *day to day*
  (`stability_across_days` re-quotes and re-fits to emulate the brief's
  "recalibrate on two different days and compare").

## Validation (`selftest.py`)

1. B&S put-call parity.
2. B&S closed-form Greeks vs finite differences.
3. Heston → B&S convergence as ξ → 0.
4. Implied-vol round-trip.
5. Calibration recovers known parameters from a synthetic smile.

## References

- Heston (1993), *A Closed-Form Solution for Options with Stochastic Volatility*.
- Black & Scholes (1973).
- Albrecher et al. (2007), *The Little Heston Trap*.
- Gatheral (2006), *The Volatility Surface*.
