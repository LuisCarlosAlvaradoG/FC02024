"""
Bloomberg-terminal visual identity for the dashboard.

A dark, dense, monospace aesthetic: near-black panels, amber primary accent,
phosphor-green for positive / longs, red for negative / shorts, cyan for the
Heston engine and grey for B&S. No emojis, no rounded toy cards — this is a
trading-desk tool. Exposes:

- COLORS / a Plotly template registered as "terminal"
- inject_css(st): Streamlit CSS override
- terminal_table(...): an HTML monospace table renderer
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- palette --------------------------------------------------------------
COLORS = {
    "bg": "#0a0e14",
    "panel": "#0f141c",
    "panel2": "#141b26",
    "grid": "#1e2733",
    "border": "#27313f",
    "text": "#c8d3e0",
    "muted": "#6b7888",
    "amber": "#ffb000",      # primary accent / market
    "green": "#3fe080",      # positive / long / calls
    "red": "#ff4d5e",        # negative / short / puts
    "cyan": "#28c8ff",       # Heston engine
    "grey": "#8a97a8",       # Black-Scholes engine
    "purple": "#b58cff",
}

ENGINE_COLOR = {"Black-Scholes": COLORS["grey"], "Heston": COLORS["cyan"],
                "Market": COLORS["amber"]}

FONT = "JetBrains Mono, IBM Plex Mono, Consolas, monospace"


def _register_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family=FONT, size=12, color=COLORS["text"]),
        colorway=[COLORS["amber"], COLORS["cyan"], COLORS["green"],
                  COLORS["red"], COLORS["purple"], COLORS["grey"]],
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"],
                   linecolor=COLORS["border"], tickcolor=COLORS["border"],
                   showspikes=True, spikecolor=COLORS["muted"],
                   spikethickness=1, spikemode="across", spikedash="dot"),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"],
                   linecolor=COLORS["border"], tickcolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"],
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=56, r=24, t=44, b=44),
        hoverlabel=dict(bgcolor=COLORS["panel2"], bordercolor=COLORS["amber"],
                        font=dict(family=FONT, color=COLORS["text"])),
        title=dict(font=dict(family=FONT, size=14, color=COLORS["amber"]),
                   x=0.01, xanchor="left"),
    )
    pio.templates["terminal"] = t
    return t


_register_template()
pio.templates.default = "terminal"


def base_layout(title=None, height=360, **kw):
    """Convenience kwargs for a consistently themed figure."""
    lay = dict(template="terminal", height=height,
               title=dict(text=title) if title else None)
    lay.update(kw)
    return lay


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:#0a0e14; --panel:#0f141c; --panel2:#141b26; --border:#27313f;
  --text:#c8d3e0; --muted:#6b7888; --amber:#ffb000; --green:#3fe080;
  --red:#ff4d5e; --cyan:#28c8ff; --grey:#8a97a8;
}
html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', Consolas, monospace !important;
}
.stApp { background: radial-gradient(circle at 50% -10%, #11161f 0%, var(--bg) 60%) !important; }
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }

/* Top command bar */
.term-header {
  display:flex; align-items:center; justify-content:space-between;
  border:1px solid var(--border); border-left:3px solid var(--amber);
  background:linear-gradient(90deg, var(--panel2), var(--panel));
  padding:10px 16px; margin-bottom:14px;
}
.term-header .title { color:var(--amber); font-weight:700; letter-spacing:2px; font-size:15px; }
.term-header .sub { color:var(--muted); font-size:11px; letter-spacing:1px; }

/* Stat tiles */
.tile-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.tile {
  flex:1; min-width:120px; border:1px solid var(--border);
  background:var(--panel); padding:8px 12px; border-top:2px solid var(--border);
}
.tile .k { color:var(--muted); font-size:10px; letter-spacing:1px; text-transform:uppercase; }
.tile .v { font-size:19px; font-weight:700; margin-top:2px; }
.tile.amber { border-top-color:var(--amber); } .tile.amber .v { color:var(--amber); }
.tile.green { border-top-color:var(--green); } .tile.green .v { color:var(--green); }
.tile.red   { border-top-color:var(--red); }   .tile.red .v   { color:var(--red); }
.tile.cyan  { border-top-color:var(--cyan); }   .tile.cyan .v  { color:var(--cyan); }
.tile.grey  { border-top-color:var(--grey); }   .tile.grey .v  { color:var(--text); }

/* Section labels */
.sec { color:var(--amber); font-size:12px; letter-spacing:2px; text-transform:uppercase;
       border-bottom:1px solid var(--border); padding-bottom:4px; margin:6px 0 10px; }

/* Monospace terminal table */
table.tt { width:100%; border-collapse:collapse; font-size:12px; }
table.tt th { color:var(--muted); text-align:right; font-weight:500;
  border-bottom:1px solid var(--border); padding:4px 10px; text-transform:uppercase; font-size:10px; }
table.tt td { text-align:right; padding:4px 10px; border-bottom:1px solid #161d27; }
table.tt td.l, table.tt th.l { text-align:left; }
table.tt tr:hover td { background:var(--panel2); }
.pos { color:var(--green); } .neg { color:var(--red); } .acc { color:var(--amber); }

/* Pills / flags */
.pill { display:inline-block; padding:2px 9px; border:1px solid var(--border);
  font-size:11px; letter-spacing:1px; }
.pill.heston { color:var(--cyan); border-color:var(--cyan); }
.pill.bs { color:var(--grey); border-color:var(--grey); }
.pill.ok { color:var(--green); border-color:var(--green); }
.pill.warn { color:var(--amber); border-color:var(--amber); }
.pill.bad { color:var(--red); border-color:var(--red); }

/* Streamlit widget polish */
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  background:var(--panel); color:var(--muted); border:1px solid var(--border);
  border-bottom:none; font-size:12px; letter-spacing:1px; text-transform:uppercase; }
.stTabs [aria-selected="true"] { color:var(--amber) !important; border-top:2px solid var(--amber); }
[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { font-family:'JetBrains Mono', monospace !important; }
.stButton>button { background:var(--panel2); color:var(--amber); border:1px solid var(--amber);
  border-radius:0; font-family:'JetBrains Mono'; letter-spacing:1px; }
.stButton>button:hover { background:var(--amber); color:#000; }
div[data-baseweb="select"]>div, .stNumberInput input, .stTextInput input {
  background:var(--panel2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:0 !important; }
</style>
"""


def inject_css(st):
    st.markdown(CSS, unsafe_allow_html=True)
