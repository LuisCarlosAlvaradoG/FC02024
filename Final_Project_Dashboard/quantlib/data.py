"""
Market-data pipeline: retrieval, cleaning, and snapshotting.

Design goals from the brief:
- Pull real option chains + spot from yfinance, a risk-free proxy from FRED.
- Aggressively clean dirty free data (zero-bid, zero-volume, crossed/wide
  spreads, stale strikes) before anything touches a model.
- Persist a single timestamped snapshot so the dashboard runs *offline* during
  the live defense, and can also pull *live* on demand.
- Degrade gracefully: if the network is unavailable, synthesise an arbitrage-
  consistent smile so the dashboard still demonstrates every feature.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timezone

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_R = 0.043          # fallback risk-free if FRED is unreachable
DAYS_PER_YEAR = 365.0

# OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits).
# e.g. VIX260819C00024000 -> VIX, 2026-08-19, Call, strike 24.000
_OCC_RE = re.compile(r"^(?P<root>[A-Z0-9.]+?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass
class ParsedContract:
    symbol: str
    root: str
    option_type: str        # "call" / "put"
    strike: float
    expiry: date


def parse_occ_symbol(symbol: str) -> ParsedContract:
    """Parse a Yahoo/OCC option symbol into its components. Raises on malformed."""
    s = symbol.strip().upper().lstrip("^").replace(" ", "")
    m = _OCC_RE.match(s)
    if not m:
        raise ValueError(
            f"'{symbol}' is not a valid option symbol. Expected ROOT+YYMMDD+"
            "C/P+8-digit-strike, e.g. VIX260819C00024000 or AAPL260116C00150000.")
    yy, mm, dd = int(m["ymd"][:2]), int(m["ymd"][2:4]), int(m["ymd"][4:6])
    expiry = date(2000 + yy, mm, dd)
    return ParsedContract(
        symbol=s, root=m["root"],
        option_type="call" if m["cp"] == "C" else "put",
        strike=int(m["strike"]) / 1000.0, expiry=expiry)


@dataclass
class MarketSnapshot:
    """One clean, self-contained market state for a single underlying."""
    ticker: str
    spot: float
    r: float
    q: float
    asof: str
    chains: dict = field(default_factory=dict)   # expiry -> cleaned DataFrame
    source: str = "yfinance"

    # ---- persistence -------------------------------------------------------
    def to_dir(self, directory=DATA_DIR):
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, f"{self.ticker.upper()}_snapshot")
        meta = {
            "ticker": self.ticker, "spot": self.spot, "r": self.r,
            "q": self.q, "asof": self.asof, "source": self.source,
            "expiries": list(self.chains.keys()),
        }
        with open(base + ".json", "w") as fh:
            json.dump(meta, fh, indent=2)
        frames = []
        for expiry, df in self.chains.items():
            tmp = df.copy()
            tmp["expiry"] = expiry
            frames.append(tmp)
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(base + ".parquet")
        return base

    @classmethod
    def from_dir(cls, ticker, directory=DATA_DIR):
        base = os.path.join(directory, f"{ticker.upper()}_snapshot")
        with open(base + ".json") as fh:
            meta = json.load(fh)
        chains = {}
        if os.path.exists(base + ".parquet"):
            allrows = pd.read_parquet(base + ".parquet")
            for expiry, df in allrows.groupby("expiry"):
                chains[expiry] = df.drop(columns=["expiry"]).reset_index(drop=True)
        return cls(
            ticker=meta["ticker"], spot=meta["spot"], r=meta["r"],
            q=meta["q"], asof=meta["asof"], chains=chains,
            source=meta.get("source", "snapshot"),
        )


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_chain(df, spot, asof, option_type, r, q,
                max_rel_spread=0.35, moneyness_band=(0.70, 1.30)):
    """
    Turn a raw yfinance calls/puts frame into a clean, model-ready frame.

    Steps: drop zero-bid / zero-volume / NaN; keep positive spreads only;
    filter by relative spread; restrict to a liquid moneyness band; compute
    mid, tau, log-moneyness; sort by strike.
    """
    df = df.copy()
    needed = {"strike", "bid", "ask"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df[df["bid"] > 0]
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) > 0]
    df = df[df["ask"] >= df["bid"]]
    df = df.dropna(subset=["strike", "bid", "ask"])
    if df.empty:
        return df

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df = df[df["rel_spread"] <= max_rel_spread]

    df["moneyness"] = df["strike"] / spot
    lo, hi = moneyness_band
    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

    expiry = pd.to_datetime(df["expiry"].iloc[0]) if "expiry" in df.columns else None
    df["option_type"] = option_type
    df["log_moneyness"] = np.log(df["strike"] / spot)
    return df.sort_values("strike").reset_index(drop=True)


def _to_naive(ts):
    """pandas Timestamp stripped of timezone, so naive/aware can be subtracted."""
    ts = pd.to_datetime(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _tau_years(expiry, asof):
    exp = _to_naive(expiry)
    now = _to_naive(asof)
    return max((exp - now).total_seconds() / (DAYS_PER_YEAR * 86400.0), 1.0 / DAYS_PER_YEAR)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def fetch_risk_free(default=DEFAULT_R):
    """13-week T-Bill (DTB3) from FRED as a continuous-comp proxy."""
    try:
        import pandas_datareader.data as web
        s = web.DataReader("DTB3", "fred").dropna()
        return float(s.iloc[-1]) / 100.0
    except Exception:
        return default


def fetch_live(ticker, n_expiries=4, default_r=DEFAULT_R):
    """Pull a live snapshot from yfinance and clean it. Raises on failure."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}")
    spot = float(hist["Close"].iloc[-1])

    try:
        info = tk.info or {}
        q = float(info.get("dividendYield") or 0.0)
        q = q / 100.0 if q > 1 else q       # yfinance sometimes returns %
    except Exception:
        q = 0.0

    r = fetch_risk_free(default_r)
    asof = datetime.now(timezone.utc).isoformat()
    expiries = list(tk.options)[:n_expiries]
    if not expiries:
        raise RuntimeError(f"No option expiries for {ticker}")

    chains = {}
    for expiry in expiries:
        oc = tk.option_chain(expiry)
        calls = oc.calls.assign(expiry=expiry)
        puts = oc.puts.assign(expiry=expiry)
        cc = clean_chain(calls, spot, asof, "call", r, q)
        cp = clean_chain(puts, spot, asof, "put", r, q)
        merged = pd.concat([cc, cp], ignore_index=True)
        if not merged.empty:
            merged["tau"] = _tau_years(expiry, asof)
            chains[expiry] = merged

    if not chains:
        raise RuntimeError(f"All chains empty after cleaning for {ticker}")
    return MarketSnapshot(ticker.upper(), spot, r, q, asof, chains, "yfinance")


# --------------------------------------------------------------------------
# Single specific contract (entered by OCC symbol)
# --------------------------------------------------------------------------
@dataclass
class ContractData:
    """One specific, real option contract priced from its own market data."""
    symbol: str
    root: str
    underlying: str
    option_type: str
    strike: float
    expiry: str
    tau: float
    spot: float
    r: float
    q: float
    bid: float
    ask: float
    mid: float                 # chosen market price (mid, else last trade)
    price_source: str          # "bid/ask mid" / "last trade" / "bid only" / ...
    iv: float                  # market IV (Yahoo) or inverted from price
    iv_source: str             # "yahoo" / "inverted" / "unavailable"
    volume: float
    open_interest: float
    history: pd.DataFrame       # the contract's own OHLC history
    asof: str
    warnings: list = field(default_factory=list)


def _spot_for(underlying, fallback=None):
    """Last price for the underlying (handles index roots like ^VIX)."""
    import yfinance as yf
    for cand in [underlying, f"^{underlying}"]:
        if not cand:
            continue
        try:
            h = yf.Ticker(cand).history(period="5d")
            if not h.empty:
                return float(h["Close"].iloc[-1]), cand
        except Exception:
            continue
    return (fallback, underlying)


def fetch_contract(symbol, default_r=DEFAULT_R, history_period="6mo"):
    """
    Retrieve ONE specific option contract by its OCC symbol: parse the symbol
    for type/strike/expiry, download the contract's own price history, pull its
    live quote, resolve the underlying spot, and derive IV (Yahoo's if present,
    otherwise inverted from the market price). Raises only if the symbol is
    unparseable or no data at all is available.
    """
    import yfinance as yf
    from .implied_vol import implied_vol

    pc = parse_occ_symbol(symbol)
    warnings = []
    tk = yf.Ticker(pc.symbol)

    # Contract's own traded history.
    try:
        hist = tk.history(period=history_period)
    except Exception:
        hist = pd.DataFrame()
    last_close = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan

    # Live quote fields.
    info = {}
    try:
        info = tk.info or {}
    except Exception as exc:
        warnings.append(f"quote info unavailable ({type(exc).__name__})")

    bid = float(info.get("bid") or 0.0)
    ask = float(info.get("ask") or 0.0)
    volume = float(info.get("volume") or (hist["Volume"].iloc[-1] if not hist.empty else 0) or 0)
    oi = float(info.get("openInterest") or 0.0)
    underlying = info.get("underlyingSymbol") or pc.root

    # Choose a market price: clean mid if possible, else last trade.
    if bid > 0 and ask > 0 and ask >= bid:
        mid, price_source = 0.5 * (bid + ask), "bid/ask mid"
    elif np.isfinite(last_close) and last_close > 0:
        mid, price_source = last_close, "last trade (dirty quote)"
        warnings.append("bid/ask crossed or zero — using last traded price")
    elif bid > 0:
        mid, price_source = bid, "bid only"
    else:
        mid, price_source = np.nan, "unavailable"
        warnings.append("no usable market price for this contract")

    # Underlying spot, rate, dividend.
    spot, used_under = _spot_for(underlying)
    if spot is None:
        raise RuntimeError(f"Could not resolve underlying spot for {underlying}")
    r = fetch_risk_free(default_r)
    q = 0.0 if str(used_under).startswith("^") else float(info.get("dividendYield") or 0.0)
    if q > 1:
        q /= 100.0

    asof = datetime.now(timezone.utc).isoformat()
    tau = _tau_years(pc.expiry.isoformat(), asof)

    # IV: prefer Yahoo's, else invert from the chosen market price.
    yiv = info.get("impliedVolatility")
    if yiv is not None and np.isfinite(yiv) and yiv > 0:
        iv, iv_source = float(yiv), "yahoo"
    elif np.isfinite(mid):
        iv = implied_vol(mid, spot, pc.strike, tau, r, q, pc.option_type)
        iv_source = "inverted" if np.isfinite(iv) else "unavailable"
        if not np.isfinite(iv):
            warnings.append("IV inversion failed (price outside no-arb bounds)")
    else:
        iv, iv_source = np.nan, "unavailable"

    return ContractData(
        symbol=pc.symbol, root=pc.root, underlying=str(used_under),
        option_type=pc.option_type, strike=pc.strike,
        expiry=pc.expiry.isoformat(), tau=tau, spot=spot, r=r, q=q,
        bid=bid, ask=ask, mid=mid, price_source=price_source,
        iv=iv, iv_source=iv_source, volume=volume, open_interest=oi,
        history=hist, asof=asof, warnings=warnings)


# --------------------------------------------------------------------------
# Synthetic fallback (offline demo / unit tests)
# --------------------------------------------------------------------------
def synthetic_snapshot(ticker="DEMO", spot=100.0, r=0.043, q=0.0,
                       expiries_days=(30, 90, 180, 365), seed=7):
    """
    Build an arbitrage-consistent smile from a known Heston parameter set, so
    every panel (smile, calibration, Greeks, portfolio) works with no network.
    """
    from .heston import HestonParams, heston_price
    from .implied_vol import implied_vol

    rng = np.random.default_rng(seed)
    true_p = HestonParams(v0=0.040, kappa=1.5, theta=0.045, xi=0.6, rho=-0.65)
    asof = datetime.now(timezone.utc).isoformat()
    chains = {}
    for d in expiries_days:
        tau = d / DAYS_PER_YEAR
        strikes = np.round(spot * np.linspace(0.80, 1.20, 17), 2)
        rows = []
        for K in strikes:
            otype = "put" if K < spot else "call"
            price = heston_price(spot, K, tau, r, true_p, q, otype)
            iv = implied_vol(price, spot, K, tau, r, q, otype)
            if not np.isfinite(iv):
                continue
            spread = max(0.02 * price, 0.01)
            mid = price * (1 + rng.normal(0, 0.002))   # tiny micro-noise
            rows.append({
                "strike": K, "bid": mid - spread / 2, "ask": mid + spread / 2,
                "mid": mid, "volume": int(rng.integers(50, 5000)),
                "openInterest": int(rng.integers(100, 20000)),
                "impliedVolatility": iv, "option_type": otype,
                "moneyness": K / spot, "log_moneyness": float(np.log(K / spot)),
                "rel_spread": spread / mid, "tau": tau,
                "expiry": (datetime.now(timezone.utc)).date().isoformat(),
            })
        df = pd.DataFrame(rows)
        chains[f"{d}d"] = df
    snap = MarketSnapshot(ticker, spot, r, q, asof, chains, "synthetic")
    snap._true_params = true_p     # exposed for the self-test
    return snap


def load_or_fetch(ticker, prefer="live", n_expiries=4):
    """
    Resolve a snapshot. prefer in {'live','snapshot','synthetic'}.
    Falls back gracefully: live -> snapshot -> synthetic.
    """
    if prefer == "synthetic":
        return synthetic_snapshot(ticker)
    if prefer == "snapshot":
        try:
            return MarketSnapshot.from_dir(ticker)
        except Exception:
            return synthetic_snapshot(ticker)
    # live
    try:
        snap = fetch_live(ticker, n_expiries=n_expiries)
        snap.to_dir()
        return snap
    except Exception:
        try:
            return MarketSnapshot.from_dir(ticker)
        except Exception:
            return synthetic_snapshot(ticker)


def chain_to_quotes(snapshot, expiries=None):
    """Flatten cleaned chains into a list of calibration quotes."""
    quotes = []
    keys = expiries or list(snapshot.chains.keys())
    for expiry in keys:
        df = snapshot.chains[expiry]
        for _, row in df.iterrows():
            iv = row.get("impliedVolatility", np.nan)
            if not np.isfinite(iv) or iv <= 0:
                continue
            quotes.append({
                "K": float(row["strike"]),
                "tau": float(row["tau"]),
                "iv": float(iv),
                "option_type": row.get("option_type", "call"),
                "mid": float(row.get("mid", np.nan)),
                # weight ATM quotes more — they anchor the level.
                "weight": 1.0 / (1.0 + 5.0 * abs(np.log(row["strike"] / snapshot.spot))),
            })
    return quotes
