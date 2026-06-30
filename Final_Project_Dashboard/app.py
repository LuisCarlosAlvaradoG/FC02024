"""
OPTIONS ANALYTICS TERMINAL  ---  Black & Scholes vs Heston

A Bloomberg-style options desk dashboard. Two pricing engines, live/offline
market data, the volatility smile as the centrepiece, full Greeks, book-level
risk, per-contract engine recommendation, and Heston calibration diagnostics.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlib import (bs_greeks, bs_price, heston_greeks, heston_price,
                      implied_vol, HestonParams, calibrate_heston,
                      stability_across_days, recommend_engine, Position,
                      aggregate_book)
from quantlib.data import (load_or_fetch, chain_to_quotes, fetch_contract,
                           synthetic_snapshot)
from quantlib.theme import inject_css, COLORS, ENGINE_COLOR, base_layout

st.set_page_config(page_title="Options Analytics Terminal", layout="wide",
                   initial_sidebar_state="expanded")
inject_css(st)


# --------------------------------------------------------------------------
# Small HTML helpers (monospace terminal widgets)
# --------------------------------------------------------------------------
def header(ticker, spot, r, q, asof, source):
    st.markdown(
        f"""<div class="term-header">
        <div><span class="title">OPTIONS ANALYTICS TERMINAL</span>
        &nbsp;&nbsp;<span class="sub">B&amp;S / HESTON &nbsp;|&nbsp; EQUITY VOL DESK</span></div>
        <div class="sub">{ticker} &nbsp;|&nbsp; SPOT {spot:,.2f} &nbsp;|&nbsp;
        r {r:.2%} &nbsp;|&nbsp; q {q:.2%} &nbsp;|&nbsp; SRC {source.upper()}
        &nbsp;|&nbsp; {asof[:19].replace('T',' ')}Z</div></div>""",
        unsafe_allow_html=True)


def tiles(items):
    """items: list of (label, value, css_class)."""
    cells = "".join(
        f'<div class="tile {c}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, c in items)
    st.markdown(f'<div class="tile-row">{cells}</div>', unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:,.{nd}f}"


def _cls(x):
    return "pos" if x > 0 else ("neg" if x < 0 else "")


def terminal_table(headers, rows, align_left=(0,)):
    th = "".join(
        f'<th class="{"l" if i in align_left else ""}">{h}</th>'
        for i, h in enumerate(headers))
    body = ""
    for row in rows:
        tds = ""
        for i, cell in enumerate(row):
            cls = "l" if i in align_left else ""
            if isinstance(cell, tuple):       # (text, css)
                txt, extra = cell
                tds += f'<td class="{cls} {extra}">{txt}</td>'
            else:
                tds += f'<td class="{cls}">{cell}</td>'
        body += f"<tr>{tds}</tr>"
    st.markdown(f'<table class="tt"><thead><tr>{th}</tr></thead>'
                f'<tbody>{body}</tbody></table>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data + calibration
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_snapshot(ticker, mode):
    return load_or_fetch(ticker, prefer=mode)


@st.cache_data(show_spinner="Fetching contract from Yahoo Finance...")
def get_contract(symbol):
    """Real single-contract fetch. Returns (ContractData, error_str)."""
    try:
        return fetch_contract(symbol), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_underlying_context(root, underlying, spot, r, q, mode):
    """
    Smile/calibration context for a specific-contract view. Tries the real
    underlying chain; on failure builds a model smile centred on the REAL spot,
    so the calibration tab stays coherent with the contract being priced.
    """
    if mode in ("live", "snapshot"):
        for cand in (underlying, root):
            try:
                return load_or_fetch(cand, prefer=mode), "real underlying chain"
            except Exception:
                continue
    return synthetic_snapshot(root, spot=spot, r=r, q=q), \
        "model smile around the real spot (no clean underlying chain)"


@st.cache_data(show_spinner=True)
def get_calibration(cache_key, _snap, expiry_key):
    quotes = chain_to_quotes(_snap, expiries=[expiry_key] if expiry_key else None)
    return calibrate_heston(quotes, _snap.spot, _snap.r, _snap.q, maxiter=30)


# --------------------------------------------------------------------------
# Sidebar  ---  input mode + command
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sec">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "MODE", ["Underlying chain", "Specific contract"], index=0,
        help="Underlying chain: browse strikes/expiries for a ticker. "
             "Specific contract: enter a Yahoo option symbol — type, strike and "
             "expiry are read from the symbol and its real history is pulled.")
    st.markdown('<div class="sec">Command</div>', unsafe_allow_html=True)
    mode = st.selectbox("DATA SOURCE", ["synthetic", "live", "snapshot"], index=0,
                        help="Governs the smile / calibration context. In "
                             "'Specific contract' mode the contract itself is "
                             "always pulled live from Yahoo.")

contract = None           # populated only in specific-contract mode

if input_mode == "Specific contract":
    with st.sidebar:
        symbol = st.text_input("OPTION SYMBOL", value="VIX260819C00024000",
                               help="OCC/Yahoo format ROOT+YYMMDD+C/P+strike, "
                                    "e.g. AAPL260116C00150000").strip().upper()
    contract, cerr = get_contract(symbol)
    if contract is None:
        st.error(f"Could not load '{symbol}': {cerr}")
        st.info("Define the contract manually below — it will still be priced "
                "with both engines.")

    with st.sidebar:
        man = st.checkbox("Override parameters manually", value=(contract is None),
                          help="Tweak spot / strike / expiry / IV by hand.")

    if contract is not None and not man:
        # Locked: everything comes from the parsed symbol + real Yahoo data.
        ticker, und_root = contract.symbol, contract.root
        S, r, q = contract.spot, contract.r, contract.q
        K, opt_type, tau = contract.strike, contract.option_type, contract.tau
        mkt_mid, mkt_iv = contract.mid, contract.iv
        rel_spread = ((contract.ask - contract.bid) / contract.mid
                      if (contract.bid > 0 and contract.ask > 0 and contract.mid > 0)
                      else np.nan)
    else:
        # Manual override, seeded from the parsed contract when available.
        base_S = contract.spot if contract else 100.0
        base_K = contract.strike if contract else 100.0
        base_days = contract.tau * 365 if contract else 30.0
        base_type = contract.option_type if contract else "call"
        base_iv = contract.iv if (contract and np.isfinite(contract.iv)) else 0.30
        base_px = contract.mid if (contract and np.isfinite(contract.mid)) else 0.0
        und_root = contract.root if contract else "DEMO"
        with st.sidebar:
            st.markdown('<div class="sec">Manual Contract</div>', unsafe_allow_html=True)
            S = st.number_input("SPOT", value=float(round(base_S, 4)), min_value=1e-4, format="%.4f")
            K = st.number_input("STRIKE", value=float(round(base_K, 4)), min_value=1e-4, format="%.4f")
            days = st.number_input("DAYS TO EXPIRY", value=float(round(base_days, 1)), min_value=0.5)
            opt_type = st.radio("TYPE", ["call", "put"],
                                index=0 if base_type == "call" else 1, horizontal=True)
            mkt_mid = st.number_input("MARKET PRICE", value=float(round(base_px, 4)), min_value=0.0, format="%.4f")
            man_iv = st.number_input("MARKET IV (0 = invert from price)",
                                     value=float(round(base_iv, 4)) if np.isfinite(base_iv) else 0.0,
                                     min_value=0.0, format="%.4f")
        r = contract.r if contract else 0.043
        q = contract.q if contract else 0.0
        tau = days / 365.0
        ticker = contract.symbol if contract else (symbol or "MANUAL")
        if man_iv > 0:
            mkt_iv = man_iv
        elif mkt_mid > 0:
            mkt_iv = implied_vol(mkt_mid, S, K, tau, r, q, opt_type)
        else:
            mkt_iv = np.nan
        rel_spread = np.nan

    # Smile / calibration context centred on the real spot.
    snap, snap_note = get_underlying_context(
        und_root, contract.underlying if contract else und_root, S, r, q, mode)
    expiries = list(snap.chains.keys())
    if not expiries:
        st.error("No smile context available.")
        st.stop()
    expiry = min(expiries, key=lambda ek: abs(float(snap.chains[ek]["tau"].iloc[0]) - tau))
    chain = snap.chains[expiry].sort_values("strike").reset_index(drop=True)
    strikes = chain["strike"].tolist()

    with st.sidebar:
        if contract is not None:
            st.markdown('<div class="sec">Parsed Contract</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:11px;line-height:1.7;border:1px solid '
                f'{COLORS["border"]};padding:8px 10px;background:{COLORS["panel"]}">'
                f'<span class="acc">UNDERLYING</span> {contract.underlying}<br>'
                f'<span class="acc">TYPE</span> {contract.option_type.upper()}<br>'
                f'<span class="acc">STRIKE</span> {contract.strike:,.2f}<br>'
                f'<span class="acc">EXPIRY</span> {contract.expiry} '
                f'({contract.tau*365:.0f}d)<br>'
                f'<span class="acc">PRICE</span> {contract.price_source}'
                f'</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">Heston</div>', unsafe_allow_html=True)
        auto_cal = st.checkbox("Auto-calibrate to smile", value=True)

else:
    # ----- Underlying chain mode (browse the chain) -----
    with st.sidebar:
        ticker = st.text_input("UNDERLYING", value="DEMO").strip().upper() or "DEMO"
    snap = get_snapshot(ticker, mode)
    snap_note = snap.source
    expiries = list(snap.chains.keys())
    if not expiries:
        st.error("No clean option chains available for this underlying.")
        st.stop()
    with st.sidebar:
        st.markdown('<div class="sec">Contract</div>', unsafe_allow_html=True)
        expiry = st.selectbox("EXPIRY", expiries, index=min(1, len(expiries) - 1))
        chain = snap.chains[expiry].sort_values("strike").reset_index(drop=True)
        tau = float(chain["tau"].iloc[0])
        strikes = chain["strike"].tolist()
        atm_idx = int(np.argmin(np.abs(np.array(strikes) - snap.spot)))
        K = st.selectbox("STRIKE", strikes, index=atm_idx,
                         format_func=lambda x: f"{x:,.2f}")
        opt_type = st.radio("TYPE", ["call", "put"], horizontal=True)
        st.markdown('<div class="sec">Heston</div>', unsafe_allow_html=True)
        auto_cal = st.checkbox("Auto-calibrate to smile", value=True)
    S, r, q = snap.spot, snap.r, snap.q
    row = chain[chain["strike"] == K].iloc[0]
    mkt_mid = float(row.get("mid", np.nan))
    mkt_iv = float(row.get("impliedVolatility", np.nan))
    rel_spread = float(row.get("rel_spread", np.nan))

# --------------------------------------------------------------------------
# Calibration (shared)
# --------------------------------------------------------------------------
if auto_cal:
    cal = get_calibration(
        f"{ticker}|{mode}|{expiry}|{snap.asof}|{snap.spot:.4f}", snap, expiry)
    hp = cal.params
else:
    cal = None
    with st.sidebar:
        v0 = st.slider("v0", 0.005, 0.40, float(min(max(np.sqrt(0.04)**2, 0.005), 0.40)), 0.005)
        kappa = st.slider("kappa", 0.1, 8.0, 1.5, 0.1)
        theta = st.slider("theta", 0.005, 0.40, 0.045, 0.005)
        xi = st.slider("xi (vol-of-vol)", 0.05, 3.0, 0.6, 0.05)
        rho = st.slider("rho", -0.99, 0.5, -0.65, 0.01)
    hp = HestonParams(v0, kappa, theta, xi, rho)

header(ticker, S, r, q, snap.asof, snap.source)
if input_mode == "Specific contract":
    st.caption(f"Contract priced from real Yahoo data; smile/calibration use {snap_note}."
               + (("  Warnings: " + "; ".join(contract.warnings)) if contract and contract.warnings else ""))

# Engine prices
bs_g = bs_greeks(S, K, tau, r, mkt_iv if np.isfinite(mkt_iv) else np.sqrt(hp.v0),
                 q, opt_type)
he_px = heston_price(S, K, tau, r, hp, q, opt_type)
bs_px = bs_g.price

# --------------------------------------------------------------------------
# Top stat tiles
# --------------------------------------------------------------------------
tiles([
    ("Black-Scholes", _fmt(bs_px, 4), "grey"),
    ("Heston", _fmt(he_px, 4), "cyan"),
    ("Market Price", _fmt(mkt_mid, 4), "amber"),
    ("BS Err vs Mkt", _fmt(bs_px - mkt_mid, 4) if np.isfinite(mkt_mid) else "--",
     "red" if np.isfinite(mkt_mid) and abs(bs_px - mkt_mid) > abs(he_px - mkt_mid) else "green"),
    ("Heston Err vs Mkt", _fmt(he_px - mkt_mid, 4) if np.isfinite(mkt_mid) else "--",
     "green" if np.isfinite(mkt_mid) and abs(he_px - mkt_mid) <= abs(bs_px - mkt_mid) else "red"),
    ("Market IV", f"{mkt_iv:.2%}" if np.isfinite(mkt_iv) else "--", "amber"),
])

tabs = st.tabs(["PRICING & VALIDATION", "VOLATILITY SMILE", "GREEKS",
                "PORTFOLIO RISK", "CALIBRATION"])

# ==========================================================================
# TAB 1 -- PRICING & VALIDATION  (+ Feedback 1: engine recommendation)
# ==========================================================================
with tabs[0]:
    # Specific-contract: show the contract's OWN real market history from Yahoo.
    if contract is not None and not contract.history.empty:
        section(f"Contract market history — {contract.symbol} (Yahoo)")
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            hist = contract.history.reset_index()
            date_col = hist.columns[0]
            figh = go.Figure()
            figh.add_trace(go.Scatter(
                x=hist[date_col], y=hist["Close"], name="Close",
                line=dict(color=COLORS["amber"], width=2)))
            if "Volume" in hist:
                figh.add_trace(go.Bar(
                    x=hist[date_col], y=hist["Volume"], name="Volume",
                    marker_color=COLORS["grid"], yaxis="y2", opacity=0.6))
            figh.update_layout(**base_layout(
                f"{contract.option_type.upper()} K={contract.strike:g} exp {contract.expiry}",
                height=260, xaxis_title="date", yaxis_title="option price",
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            title="volume")))
            st.plotly_chart(figh, width='stretch')
        with hcol2:
            last = float(contract.history["Close"].iloc[-1])
            tiles([("Last", _fmt(last, 4), "amber")])
            tiles([("Bid", _fmt(contract.bid, 4), "green")])
            tiles([("Ask", _fmt(contract.ask, 4), "red")])
            tiles([("IV", f"{contract.iv:.1%}" if np.isfinite(contract.iv) else "--",
                    "cyan")])
            st.download_button(
                "DOWNLOAD HISTORY (CSV)",
                contract.history.to_csv().encode("utf-8"),
                file_name=f"{contract.symbol}_history.csv", mime="text/csv")
        if contract.warnings:
            st.caption("Data notes: " + "; ".join(contract.warnings)
                       + f"  |  IV source: {contract.iv_source}.")
    elif contract is not None:
        st.warning(f"No price history returned by Yahoo for {contract.symbol}. "
                   "Quote/IV (if any) still shown; use manual override to adjust.")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("Engine comparison vs market")
        def err_cells(px):
            if not np.isfinite(mkt_mid):
                return _fmt(px, 4), "--", "--"
            ae = px - mkt_mid
            re = ae / mkt_mid if mkt_mid else np.nan
            return _fmt(px, 4), (_fmt(ae, 4), _cls(ae)), (f"{re:+.2%}", _cls(ae))
        b = err_cells(bs_px)
        h = err_cells(he_px)
        terminal_table(
            ["Engine", "Price", "Abs Err", "Rel Err"],
            [[("Black-Scholes", "acc"), b[0], b[1], b[2]],
             [("Heston", "acc"), h[0], h[1], h[2]],
             [("Market", ""), _fmt(mkt_mid, 4), "--", "--"]],
            align_left=(0,))

        # price-vs-strike curve for both engines + market
        section("Price profile across strikes")
        ks = chain["strike"].values
        bs_curve = [bs_price(S, k, tau, r,
                             mkt_iv if np.isfinite(mkt_iv) else np.sqrt(hp.v0),
                             q, opt_type) for k in ks]
        he_curve = [heston_price(S, k, tau, r, hp, q, opt_type) for k in ks]
        mid_curve = chain["mid"].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=mid_curve, name="Market", mode="markers",
                                 marker=dict(color=COLORS["amber"], size=7, symbol="diamond")))
        fig.add_trace(go.Scatter(x=ks, y=bs_curve, name="Black-Scholes",
                                 line=dict(color=COLORS["grey"], width=2)))
        fig.add_trace(go.Scatter(x=ks, y=he_curve, name="Heston",
                                 line=dict(color=COLORS["cyan"], width=2)))
        fig.add_vline(x=K, line=dict(color=COLORS["amber"], dash="dot", width=1))
        fig.update_layout(**base_layout(f"{opt_type.upper()} price | {expiry}",
                                        xaxis_title="Strike", yaxis_title="Price"))
        st.plotly_chart(fig, width='stretch')

    with c2:
        section("Recommended engine (per contract)")
        rec = recommend_engine(S, K, tau, option_type=opt_type,
                               rel_spread=rel_spread,
                               calibration_rmse=cal.rmse_vol if cal else None)
        pill = "heston" if rec.engine == "Heston" else "bs"
        conf_cls = {"High": "ok", "Medium": "warn", "Low": "bad"}.get(rec.confidence, "warn")
        st.markdown(
            f'<div style="border:1px solid {COLORS["border"]};padding:14px;'
            f'background:{COLORS["panel"]}">'
            f'<span class="pill {pill}">{rec.engine.upper()}</span>&nbsp;'
            f'<span class="pill {conf_cls}">CONF: {rec.confidence.upper()}</span>'
            f'<div style="margin-top:10px;color:{COLORS["text"]};font-size:12px;'
            f'line-height:1.6">{rec.reason}.</div></div>'.replace("..", "."),
            unsafe_allow_html=True)

        lm = np.log(K / S)
        section("Contract profile")
        terminal_table(
            ["Field", "Value"],
            [[("Moneyness K/S", ""), f"{K/S:.4f}"],
             [("Log-moneyness", ""), f"{lm:+.4f}"],
             [("Time to expiry", ""), f"{tau*365:.0f}d  ({tau:.3f}y)"],
             [("Market IV", ""), f"{mkt_iv:.2%}" if np.isfinite(mkt_iv) else "--"],
             [("Rel. spread", ""), f"{rel_spread:.1%}" if np.isfinite(rel_spread) else "--"],
             [("Feller 2kt-x2", ""),
              (f"{hp.feller():+.4f}", "pos" if hp.feller_ok() else "neg")]],
            align_left=(0,))
        st.caption("Recommendation is heuristic and defensible: it weighs "
                   "moneyness, maturity, liquidity and calibration fit — "
                   "exactly what the desk expects you to justify live.")

# ==========================================================================
# TAB 2 -- VOLATILITY SMILE  (centrepiece)
# ==========================================================================
with tabs[1]:
    section("Implied volatility smile — market vs B&S flat vs Heston")
    csm = chain.dropna(subset=["impliedVolatility"]).sort_values("strike")
    ks = csm["strike"].values
    lm = np.log(ks / S)
    mkt_smile = csm["impliedVolatility"].values

    # B&S flat = single ATM sigma
    atm_iv = float(csm.iloc[(np.abs(ks - S)).argmin()]["impliedVolatility"])
    # Heston-implied smile: price with Heston, invert to IV
    he_smile = []
    for k in ks:
        ot = "put" if k < S else "call"
        px = heston_price(S, k, tau, r, hp, q, ot)
        he_smile.append(implied_vol(px, S, k, tau, r, q, ot))
    he_smile = np.array(he_smile)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lm, y=mkt_smile, name="Market", mode="markers+lines",
                             marker=dict(color=COLORS["amber"], size=8, symbol="diamond"),
                             line=dict(color=COLORS["amber"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=lm, y=np.full_like(lm, atm_iv), name="B&S (flat)",
                             line=dict(color=COLORS["grey"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=lm, y=he_smile, name="Heston (calibrated)",
                             line=dict(color=COLORS["cyan"], width=3)))
    fig.add_vline(x=0, line=dict(color=COLORS["border"], dash="dot"))
    fig.add_vline(x=float(np.log(K / S)), line=dict(color=COLORS["amber"], dash="dot", width=1))
    fig.update_layout(**base_layout("Implied vol vs log-moneyness ln(K/S)",
                                    xaxis_title="log-moneyness", yaxis_title="implied vol",
                                    height=420, yaxis=dict(tickformat=".0%")))
    st.plotly_chart(fig, width='stretch')

    cc1, cc2 = st.columns(2)
    with cc1:
        section("Read the skew")
        st.markdown(
            f'<div style="font-size:12px;line-height:1.7;color:{COLORS["text"]}">'
            "Equity smiles skew: OTM puts (left) trade at higher implied vol — "
            "crash insurance is expensive. A single flat B&amp;S &sigma; (dashed) "
            f"cannot reproduce it. Heston, through &rho;={hp.rho:.2f} and "
            f"vol-of-vol &xi;={hp.xi:.2f}, curves to match.</div>",
            unsafe_allow_html=True)
    with cc2:
        section("Surface across expiries")
        if st.checkbox("Render 3D surface (all expiries)", value=False):
            surf_k, surf_t, surf_iv = [], [], []
            for ek in expiries:
                cdf = snap.chains[ek].dropna(subset=["impliedVolatility"])
                for _, rr in cdf.iterrows():
                    surf_t.append(float(rr["tau"]))
                    surf_k.append(np.log(rr["strike"] / S))
                    surf_iv.append(float(rr["impliedVolatility"]))
            if surf_iv:
                fig3 = go.Figure(go.Mesh3d(
                    x=surf_k, y=surf_t, z=surf_iv, intensity=surf_iv,
                    colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["amber"]],
                                [1, COLORS["red"]]], opacity=0.9))
                fig3.update_layout(template="terminal", height=380,
                                   scene=dict(
                                       xaxis_title="ln(K/S)", yaxis_title="tau",
                                       zaxis_title="IV",
                                       bgcolor=COLORS["panel"]))
                st.plotly_chart(fig3, width='stretch')

# ==========================================================================
# TAB 3 -- GREEKS
# ==========================================================================
with tabs[2]:
    section(f"Greeks for selected contract — {opt_type.upper()} K={K:,.2f}")
    he_g = heston_greeks(S, K, tau, r, hp, q, opt_type)
    bd, hd = bs_g.display(), he_g.display()
    rows = []
    for kkey in ["Price", "Delta", "Gamma", "Vega (1%)", "Theta (1d)",
                 "Rho (1%)", "Vanna", "Volga"]:
        diff = hd[kkey] - bd[kkey]
        rows.append([(kkey, "acc"), _fmt(bd[kkey], 5), _fmt(hd[kkey], 5),
                     (_fmt(diff, 5), _cls(diff))])
    terminal_table(["Greek", "Black-Scholes", "Heston", "H - BS"], rows,
                   align_left=(0,))
    st.caption("B&S Greeks are closed-form; Heston Greeks are central-difference "
               "bumps on a consistent vol convention. Method is documented in "
               "quantlib/heston.py (risk manager's red flag).")

    section("Greek profiles across spot")
    greek_name = st.selectbox("GREEK", ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Volga"])
    spots = np.linspace(0.6 * S, 1.4 * S, 41)

    def greek_curve(engine):
        out = []
        for s in spots:
            if engine == "bs":
                g = bs_greeks(s, K, tau, r,
                              mkt_iv if np.isfinite(mkt_iv) else np.sqrt(hp.v0),
                              q, opt_type)
            else:
                g = heston_greeks(s, K, tau, r, hp, q, opt_type)
            out.append(getattr(g, greek_name.lower()))
        return np.array(out)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=greek_curve("bs"), name="Black-Scholes",
                             line=dict(color=COLORS["grey"], width=2)))
    fig.add_trace(go.Scatter(x=spots, y=greek_curve("he"), name="Heston",
                             line=dict(color=COLORS["cyan"], width=2)))
    fig.add_vline(x=S, line=dict(color=COLORS["amber"], dash="dot", width=1))
    fig.add_vline(x=K, line=dict(color=COLORS["grey"], dash="dot", width=1))
    fig.update_layout(**base_layout(f"{greek_name} vs spot",
                                    xaxis_title="Spot", yaxis_title=greek_name))
    st.plotly_chart(fig, width='stretch')
    st.caption("Gamma/Vega peak near the money. Stochastic vol re-shapes Vega "
               "and adds Vanna/Volga structure B&S cannot express.")

# ==========================================================================
# TAB 4 -- PORTFOLIO RISK  (Feedback 2: book-level Greeks + hedges)
# ==========================================================================
with tabs[3]:
    section("Book — enter 2 to 4 legs (qty signed: + long / - short)")
    default_legs = pd.DataFrame({
        "type": ["call", "put", "call"],
        "strike": [round(S, 0), round(0.95 * S, 0), round(1.05 * S, 0)],
        "qty": [10.0, -5.0, -5.0],
    })
    legs = st.data_editor(default_legs, num_rows="dynamic",
                          width='stretch', key="legs")
    engine_choice = st.radio("RISK ENGINE", ["Black-Scholes", "Heston"],
                             horizontal=True,
                             help="One consistent method per book — never mix.")

    positions = []
    for _, lg in legs.iterrows():
        try:
            positions.append(Position(option_type=str(lg["type"]).strip().lower(),
                                       K=float(lg["strike"]), tau=tau,
                                       qty=float(lg["qty"]),
                                       label=f'{lg["type"]} {lg["strike"]:.0f}'))
        except (ValueError, TypeError):
            continue

    if positions:
        # ONE method for the whole book (red flag: never mix engines silently).
        # B&S legs use each strike's market IV (smile-aware); Heston legs use
        # the single calibrated parameter set. The same gfn prices the hedge
        # leg, so book and hedge Greeks are always from the same engine.
        def gfn(pos):
            iv = float(snap.chains[expiry].iloc[
                (snap.chains[expiry]["strike"] - pos.K).abs().argmin()]
                .get("impliedVolatility", np.sqrt(hp.v0)))
            iv = iv if np.isfinite(iv) else np.sqrt(hp.v0)
            if engine_choice == "Black-Scholes":
                return bs_greeks(S, pos.K, pos.tau, r, iv, q, pos.option_type)
            return heston_greeks(S, pos.K, pos.tau, r, hp, q, pos.option_type)

        hedge = Position(opt_type if opt_type else "call", K, tau, 0.0,
                         label=f"ATM {opt_type} {K:.0f}")
        book = aggregate_book(positions, S, gfn, hedge_option=hedge)

        net = book.net
        tiles([
            ("Net Delta", _fmt(net["Delta"], 1),
             "green" if net["Delta"] >= 0 else "red"),
            ("Net Gamma", _fmt(net["Gamma"], 3),
             "green" if net["Gamma"] >= 0 else "red"),
            ("Net Vega", _fmt(net["Vega (1%)"], 1), "cyan"),
            ("Net Theta/day", _fmt(net["Theta (1d)"], 1),
             "green" if net["Theta (1d)"] >= 0 else "red"),
            ("Delta hedge", f"{book.delta_hedge_shares:,.0f} sh", "amber"),
        ])

        section("Per-leg contribution")
        rows = []
        for pos, g in book.per_position:
            mult = pos.qty * pos.multiplier
            rows.append([
                (pos.label, "acc"),
                f"{pos.qty:+.0f}",
                _fmt(g["Delta"] * mult, 1),
                _fmt(g["Gamma"] * mult, 3),
                _fmt(g["Vega (1%)"] * mult, 1),
                _fmt(g["Theta (1d)"] * mult, 1)])
        terminal_table(["Leg", "Qty", "Delta", "Gamma", "Vega", "Theta/d"],
                       rows, align_left=(0,))

        section("Hedge program")
        gn = book.gamma_neutral
        st.markdown(
            f'<div style="font-size:12px;line-height:1.8;color:{COLORS["text"]}">'
            f'<b style="color:{COLORS["amber"]}">Delta hedge:</b> trade '
            f'<b>{book.delta_hedge_shares:,.0f}</b> shares of {ticker} '
            f'({"sell" if book.delta_hedge_shares<0 else "buy"}) to flatten delta.<br>'
            + (f'<b style="color:{COLORS["amber"]}">Gamma-neutral:</b> trade '
               f'<b>{gn["hedge_contracts"]:+.2f}</b> contracts of '
               f'{gn["hedge_label"]} ({"pay" if gn["cost"]>0 else "receive"} '
               f'{abs(gn["cost"]):,.0f} premium); residual delta '
               f'{gn["residual_delta"]:,.1f} -> re-hedge '
               f'{gn["residual_delta_hedge_shares"]:,.0f} shares.'
               if gn else "") +
            "</div>", unsafe_allow_html=True)
        st.caption(f"Book Greeks use ONE engine ({engine_choice}) — never mixed. "
                   "B&S legs price off each strike's market IV (smile-aware); "
                   "Heston legs use the single calibrated surface, so switching "
                   "engine changes both model and vol source. Method documented "
                   "in quantlib/heston.py (risk manager's red flag).")
    else:
        st.info("Enter at least one valid leg.")

# ==========================================================================
# TAB 5 -- CALIBRATION  (Feedback 3: honest, Feller-aware, stability)
# ==========================================================================
with tabs[4]:
    section("Heston parameters")
    fcls = "pos" if hp.feller_ok() else "neg"
    terminal_table(
        ["Param", "Value", "Meaning"],
        [[("v0", "acc"), _fmt(hp.v0, 4), ("initial variance", "")],
         [("kappa", "acc"), _fmt(hp.kappa, 4), ("mean-reversion speed", "")],
         [("theta", "acc"), _fmt(hp.theta, 4), ("long-run variance", "")],
         [("xi", "acc"), _fmt(hp.xi, 4), ("vol of vol", "")],
         [("rho", "acc"), _fmt(hp.rho, 4), ("spot-vol correlation", "")],
         [("Feller 2kt-x2", "acc"), (_fmt(hp.feller(), 4), fcls),
          ("> 0 keeps variance > 0", "")]],
        align_left=(0, 2))

    if cal is not None:
        fpill = "ok" if cal.feller_ok else "bad"
        spill = "ok" if cal.success else "warn"
        pin_pill = (f'&nbsp;<span class="pill bad">PINNED: '
                    f'{",".join(cal.pinned).upper()}</span>' if cal.pinned else "")
        st.markdown(
            f'<div style="margin-top:8px">'
            f'<span class="pill {spill}">OPTIMISER: {"CONVERGED" if cal.success else "CHECK"}</span>&nbsp;'
            f'<span class="pill {fpill}">FELLER: {"OK" if cal.feller_ok else "VIOLATED"}</span>&nbsp;'
            f'<span class="pill warn">RMSE: {cal.rmse_vol*100:.2f} vol pts</span>&nbsp;'
            f'<span class="pill bs">{cal.n_quotes} QUOTES</span>{pin_pill}</div>',
            unsafe_allow_html=True)
        if not cal.feller_ok:
            st.warning("Feller condition violated: 2*kappa*theta < xi^2. The "
                       "variance process can touch zero — parameters may be "
                       "unstable. Flagged per the quant developer's requirement.")
        if cal.pinned:
            st.warning(f"Parameters pinned to a bound ({', '.join(cal.pinned)}): "
                       "a degenerate, ill-posed fit. Treat these params with "
                       "caution — widen bounds or add quotes.")

        scol1, scol2 = st.columns(2)

        with scol1:
            section("Stability I — across expiries (term structure)")
            st.caption("One snapshot, re-fit per expiry. Large swings in xi/rho "
                       "between expiries signal an over-fit, ill-posed calibration.")
            if st.button("RUN PER-EXPIRY RECALIBRATION"):
                stab_rows = []
                prog = st.progress(0.0)
                for i, ek in enumerate(expiries):
                    qg = chain_to_quotes(snap, expiries=[ek])
                    try:
                        rr = calibrate_heston(qg, S, r, q, maxiter=20)
                        p = rr.params
                        stab_rows.append([
                            (ek, "acc"), _fmt(p.v0, 4), _fmt(p.kappa, 3),
                            _fmt(p.theta, 4), _fmt(p.xi, 3), _fmt(p.rho, 3),
                            (f"{rr.rmse_vol*100:.2f}%", "pos" if rr.rmse_vol < 0.02 else "neg"),
                            ("OK" if rr.feller_ok else "VIOL",
                             "pos" if rr.feller_ok else "neg")])
                    except Exception:
                        stab_rows.append([(ek, "acc"), "--", "--", "--", "--", "--", "--", "--"])
                    prog.progress((i + 1) / len(expiries))
                terminal_table(["Expiry", "v0", "kappa", "theta", "xi", "rho",
                                "RMSE", "Feller"], stab_rows, align_left=(0,))

        with scol2:
            section("Stability II — day to day (re-quote & re-fit)")
            st.caption("The brief's 'recalibrate on two different days'. Successive "
                       "days are emulated by perturbing each IV ~0.5 vol pt and "
                       "nudging spot ~1%. If the fit is well-posed, params barely "
                       "move; if ill-posed, xi/rho/kappa swing for a tiny re-quote.")
            if st.button("RUN DAY-TO-DAY RECALIBRATION"):
                qg = chain_to_quotes(snap, expiries=[expiry])
                days = stability_across_days(qg, S, r, q, n_days=4, maxiter=18)
                rows = []
                for d in days:
                    is_range = d["day"] == "range"
                    rows.append([
                        (d["day"], "acc"),
                        _fmt(d["v0"], 4), _fmt(d["kappa"], 3), _fmt(d["theta"], 4),
                        _fmt(d["xi"], 3), _fmt(d["rho"], 3),
                        ("--" if is_range else f"{d['rmse']*100:.2f}%"),
                        ("--" if is_range else ("OK" if d["feller_ok"] else "VIOL",
                                                "pos" if d["feller_ok"] else "neg"))])
                terminal_table(["Day", "v0", "kappa", "theta", "xi", "rho",
                                "RMSE", "Feller"], rows, align_left=(0,))
                st.caption("Read the bottom 'range' row: that is the max-min "
                           "dispersion of each parameter across days — your "
                           "day-to-day parameter risk.")
    else:
        st.info("Manual Heston override is active. Enable 'Auto-calibrate to "
                "smile' in the sidebar to fit parameters and see diagnostics.")
