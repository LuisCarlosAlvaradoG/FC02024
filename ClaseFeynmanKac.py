"""
==========================================================================
  QUANTITATIVE FINANCE — ITESO  |  L.C. Alvarado
  Clase: Movimiento Browniano → Feynman–Kač → Black–Scholes
==========================================================================

MÓDULOS (ejecutar en orden narrativo):
  1. bm_single_particle()      — 1 partícula: la caminata aleatoria pura
  2. bm_diffusion()            — N partículas: dispersión tipo "tintan en agua"
  3. bm_1d_to_stock()          — De W(t) a precio de acción (2-D horizontal)
  4. gbm_drift_comparison()    — ★ NUEVO: mismo σ, distintos μ (y drift=r bajo Q)
  5. montecarlo_convergence()  — Precio MC vs. fórmula cerrada B–S
  6. full_story()              — Panel único 2×3: toda la narrativa de una pasada

Correr todo:
  python ClaseFeynmanKac.py
O un módulo individual:
  python -c "from ClaseFeynmanKac import gbm_drift_comparison; gbm_drift_comparison()"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.stats import norm
from matplotlib.lines import Line2D

# ── Paleta global ──────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_C    = "#21262d"
WHITE     = "#e6edf3"
CYAN      = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#f0883e"
RED       = "#f85149"
PURPLE    = "#bc8cff"
YELLOW    = "#e3b341"
PINK      = "#ff7b72"

ACCENT    = [CYAN, GREEN, ORANGE, RED, PURPLE, YELLOW, PINK]

def _style_dark(fig, *axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.grid(color=GRID_C, linewidth=0.6)


# ==========================================================================
# 1. BM — Una sola partícula (1-D, animada)
# ==========================================================================
def bm_single_particle(save_gif=False):
    T, dt, sigma = 100, 0.01, 1.0
    t     = np.linspace(0, T * dt, T)
    T_end = t[-1]
    y_max =  3 * sigma * np.sqrt(T_end)
    y_min = -y_max

    # W mutable — se regenera en cada loop
    W = np.zeros(T)

    def new_path():
        """Genera un camino browniano nuevo in-place."""
        W[0] = 0.0
        W[1:] = np.cumsum(sigma * np.sqrt(dt) * np.random.randn(T - 1))

    new_path()   # camino inicial al abrir la ventana

    fig, ax = plt.subplots(figsize=(10, 4))
    _style_dark(fig, ax)
    ax.set_xlim(t[0], T_end)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Tiempo  t")
    ax.set_ylabel("W(t)")
    ax.set_title("Movimiento Browniano  —  Una sola partícula", color=WHITE, fontsize=13)
    ax.axhline(0, color=GRID_C, lw=1)

    band_t = np.linspace(0.001, T_end, 300)
    ax.fill_between(band_t,
                    -sigma * np.sqrt(band_t),  sigma * np.sqrt(band_t),
                    alpha=0.14, color=CYAN, label=r"$\pm\sigma\sqrt{t}$  (1σ)")
    ax.fill_between(band_t,
                    -2 * sigma * np.sqrt(band_t), 2 * sigma * np.sqrt(band_t),
                    alpha=0.06, color=CYAN, label=r"$\pm2\sigma\sqrt{t}$  (2σ)")
    ax.fill_between(band_t,
                    -3 * sigma * np.sqrt(band_t), 3 * sigma * np.sqrt(band_t),
                    alpha=0.04, color=CYAN, label=r"$\pm3\sigma\sqrt{t}$  (3σ)")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    trail_len    = 120
    FLASH_FRAMES = 18
    TOTAL        = T + FLASH_FRAMES

    lc_obj = ax.add_collection(LineCollection([], linewidths=2))
    head,  = ax.plot([], [], "o", ms=7,  color=WHITE, zorder=5)
    flash, = ax.plot([], [], "o", ms=22, color=RED,   zorder=6, alpha=0.0,
                     markeredgecolor=RED)
    txt    = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                     color=WHITE, fontsize=10)

    def update(f):
        if f < T:
            s      = max(0, f - trail_len)
            xv, yv = t[s:f+1], W[s:f+1]
            if len(xv) >= 2:
                segs   = np.stack([np.column_stack([xv[:-1], yv[:-1]]),
                                   np.column_stack([xv[1:],  yv[1:]])], axis=1)
                n      = len(segs)
                alphas = np.linspace(0.05, 1.0, n)
                rgba   = np.zeros((n, 4))
                rgba[:, 0] = 0.34; rgba[:, 1] = 0.65; rgba[:, 2] = 1.0
                rgba[:, 3] = alphas
                lc_obj.set_segments(segs)
                lc_obj.set_color(rgba)
            head.set_data([t[f]], [W[f]])
            flash.set_alpha(0.0)
            txt.set_text(f"t = {t[f]:.2f}    W(t) = {W[f]:.3f}")
        else:
            fi    = f - T
            pulse = np.sin(np.pi * fi / FLASH_FRAMES) ** 2
            flash.set_data([t[-1]], [W[-1]])
            flash.set_alpha(pulse)
            head.set_alpha(1.0 - 0.5 * pulse)
            txt.set_text(f"t = T = {T_end:.2f}    W(T) = {W[-1]:.3f}  ← fin")

        return lc_obj, head, flash, txt

    def on_repeat():
        """Se llama automáticamente al inicio de cada loop — nuevo camino."""
        new_path()
        lc_obj.set_segments([])   # limpia estela del loop anterior
        head.set_alpha(1.0)

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=18,
                         blit=True, repeat=True)
    anim._stop = lambda *args: None          # evita que el GC la destruya
    fig.canvas.mpl_connect('draw_event',
        lambda e: None)                      # mantiene canvas vivo

    # Enganchar el repeat: FuncAnimation llama a repeat_func internamente
    anim._repeat_delay = 0
    # El hook más limpio disponible en matplotlib ≥ 3.5:
    anim.new_frame_seq = lambda: (on_repeat() or iter(range(TOTAL)))

    if save_gif:
        anim.save("01_bm_single.gif", writer=PillowWriter(fps=30))
        print("Guardado: 01_bm_single.gif")

    plt.tight_layout()
    plt.show()
# ==========================================================================
# 2. BM — Difusión de N partículas (2-D, estático final + heatmap)
# ==========================================================================
def bm_diffusion(save_fig=True):
    """
    N partículas parten del origen.  Panel izquierdo: trayectorias.
    Panel derecho: densidad final comparada con N(0, σ²T).
    """
    np.random.seed(7)
    N, T, dt, sigma = 200, 300, 0.01, 1.0
    t  = np.linspace(0, T*dt, T)
    dW = sigma * np.sqrt(dt) * np.random.randn(N, T)
    W  = np.cumsum(dW, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios":[2.5,1]})
    _style_dark(fig, ax1, ax2)

    # — Trayectorias —
    cmap = plt.cm.cool
    for i in range(N):
        ax1.plot(t, W[i], lw=0.6, alpha=0.35,
                 color=cmap(i/N))
    ax1.set_xlabel("Tiempo  t");  ax1.set_ylabel("W(t)")
    ax1.set_title(f"Difusión de {N} partículas  (σ={sigma})", fontsize=12)
    # Banda ±σ√t
    ax1.fill_between(t, -sigma*np.sqrt(t), sigma*np.sqrt(t),
                     alpha=0.15, color=CYAN, label=r"$\pm\sigma\sqrt{t}$")
    ax1.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    # — Histograma de posiciones finales vs. teórico —
    final = W[:, -1]
    ax2.hist(final, bins=25, density=True, orientation="horizontal",
             color=CYAN, alpha=0.6, edgecolor=DARK_BG, label="Simulado")
    ygrid = np.linspace(final.min()-0.5, final.max()+0.5, 300)
    theory_std = sigma * np.sqrt(T*dt)
    ax2.plot(norm.pdf(ygrid, 0, theory_std), ygrid,
             color=ORANGE, lw=2, label=rf"$\mathcal{{N}}(0,{theory_std:.2f}^2)$")
    ax2.set_xlabel("Densidad");  ax2.set_ylabel("Posición final W(T)")
    ax2.set_title("Distribución final")
    ax2.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=8)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    fig.suptitle("Movimiento Browniano — Dispersión  'tinta en agua'",
                 color=WHITE, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_fig:
        plt.savefig("02_bm_diffusion.png", dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print("Guardado: 02_bm_diffusion.png")
    plt.show()


# ==========================================================================
# 3. De W(t) a precio de acción (GBM sin drift vs con drift)
# ==========================================================================
def bm_1d_to_stock(save_fig=True):
    """
    Columna izq: trayectorias W(t) puras  (Brownian Motion)
    Columna der: S(t) = S0 · exp((μ-σ²/2)t + σ W(t))  (GBM)
    Muestra la transformación log-normal visualmente.
    """
    np.random.seed(3)
    N, T_end, dt = 60, 2.0, 1/252
    S0, mu, sigma = 100, 0.08, 0.22
    steps = int(T_end / dt)
    t = np.linspace(0, T_end, steps+1)

    dW = np.sqrt(dt) * np.random.randn(N, steps)
    W  = np.zeros((N, steps+1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    S = S0 * np.exp((mu - 0.5*sigma**2)*t[None,:] + sigma*W)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _style_dark(fig, ax1, ax2)

    cmap = plt.cm.plasma
    for i in range(N):
        c = cmap(i/N)
        ax1.plot(t, W[i], lw=0.8, alpha=0.5, color=c)
        ax2.plot(t, S[i], lw=0.8, alpha=0.5, color=c)

    # Media teórica
    ax1.axhline(0, color=WHITE, lw=1.5, ls="--", label="E[W(t)] = 0")
    ax2.plot(t, S0*np.exp(mu*t), color=ORANGE, lw=2.2, ls="--",
             label=rf"$S_0 e^{{\mu t}}$ — media teórica")

    ax1.set_title("Movimiento Browniano  W(t)", fontsize=12)
    ax1.set_xlabel("t");  ax1.set_ylabel("W(t)")
    ax1.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    ax2.set_title(r"Precio de acción  $S_t = S_0 e^{(\mu-\frac{\sigma^2}{2})t+\sigma W_t}$",
                  fontsize=12)
    ax2.set_xlabel("t");  ax2.set_ylabel("$S_t$")
    ax2.axhline(S0, color=GRID_C, lw=0.8, ls=":")
    ax2.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    fig.suptitle(r"De $W(t)$ a precio de acción  — transformación log-normal  (GBM)",
                 color=WHITE, fontsize=14)
    plt.tight_layout()
    if save_fig:
        plt.savefig("03_bm_to_stock.png", dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print("Guardado: 03_bm_to_stock.png")
    plt.show()


# ==========================================================================
# 4. ★ NUEVO — Comparación de drifts: mundo real P vs mundo Q
# ==========================================================================
def gbm_drift_comparison(save_fig=True):
    """
    Misma volatilidad σ, distintos drifts μ.
    El punto central: bajo la medida neutral al riesgo Q, μ → r.
    Tres columnas:
      A) μ > r  (acción "creciente")
      B) μ = r  (mundo Q, drift = tasa libre de riesgo)  ← La clave
      C) μ < r  (acción "decreciente")
    Panel inferior: distribución de S_T para los tres casos.
    """
    np.random.seed(42)
    S0    = 100.0
    sigma = 0.22
    r     = 0.05          # tasa libre de riesgo
    T     = 1.0
    dt    = 1/252
    steps = int(T/dt)
    N     = 80            # trayectorias por panel
    t     = np.linspace(0, T, steps+1)

    drifts = {
        rf"$\mu = 0.18$  (mundo $\mathbb{{P}}$,  acción alcista)": 0.18,
        rf"$\mu = r = 0.05$  (mundo $\mathbb{{Q}}$,  neutro al riesgo)": r,
        rf"$\mu = -0.08$  (mundo $\mathbb{{P}}$,  acción bajista)": -0.08,
    }
    colors_drift = [GREEN, CYAN, RED]
    labels_short  = [r"$\mu=0.18$", r"$\mu=r=0.05$  ← $\mathbb{Q}$", r"$\mu=-0.08$"]

    fig = plt.figure(figsize=(15, 9))
    _style_dark(fig)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                             height_ratios=[1.6, 1])

    ax_paths = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax_dist  = [fig.add_subplot(gs[1, k]) for k in range(3)]

    for (label, mu), col, ax_p, ax_d, lbl in zip(
            drifts.items(), colors_drift, ax_paths, ax_dist, labels_short):

        _style_dark(fig, ax_p, ax_d)

        # ── Simular trayectorias ──
        dW = np.sqrt(dt) * np.random.randn(N, steps)
        W  = np.zeros((N, steps+1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        paths = S0 * np.exp((mu - 0.5*sigma**2)*t[None,:] + sigma*W)

        # ── Panel superior: trayectorias ──
        for i in range(N):
            ax_p.plot(t, paths[i], lw=0.7, alpha=0.30, color=col)

        # Media teórica
        ax_p.plot(t, S0*np.exp(mu*t), color=WHITE, lw=2.2, ls="--",
                  label=rf"$S_0 e^{{\mu t}}$", zorder=5)
        ax_p.axhline(S0, color=GRID_C, lw=0.8, ls=":")
        ax_p.set_title(label, color=col, fontsize=10, pad=6)
        ax_p.set_xlabel("t",  fontsize=9)
        ax_p.set_ylabel("$S_t$", fontsize=9)
        ax_p.legend(facecolor=DARK_BG, edgecolor=GRID_C,
                    labelcolor=WHITE, fontsize=8)

        # Anota valor esperado de S_T
        E_ST = S0*np.exp(mu*T)
        ax_p.annotate(f"E[$S_T$] = {E_ST:.0f}",
                      xy=(T, E_ST), xytext=(0.68, 0.85),
                      textcoords="axes fraction",
                      color=WHITE, fontsize=9,
                      arrowprops=dict(arrowstyle="->", color=WHITE, lw=1))

        # ── Panel inferior: histograma de S_T ──
        S_T_mc = paths[:, -1]
        N_big  = 5000
        dW_big = np.sqrt(T) * np.random.randn(N_big)
        S_T_big = S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*dW_big)

        ax_d.hist(S_T_big, bins=40, density=True, color=col,
                  alpha=0.5, edgecolor=DARK_BG, label="MC (5k)")

        # Teórico log-normal
        x_grid = np.linspace(S_T_big.min(), S_T_big.max(), 300)
        log_mu   = np.log(S0) + (mu - 0.5*sigma**2)*T
        log_sig  = sigma*np.sqrt(T)
        pdf_ln   = norm.pdf(np.log(x_grid), log_mu, log_sig) / x_grid
        ax_d.plot(x_grid, pdf_ln, color=WHITE, lw=2)

        ax_d.axvline(S0*np.exp(mu*T), color=ORANGE, lw=1.8, ls="--",
                     label=f"E[S_T]={S0*np.exp(mu*T):.0f}")
        ax_d.axvline(S0, color=GRID_C, lw=1.0, ls=":")
        ax_d.set_xlabel("$S_T$", fontsize=9)
        ax_d.set_ylabel("Densidad", fontsize=9)
        ax_d.set_title(f"Distribución de $S_T$  —  {lbl}", fontsize=9)
        ax_d.legend(facecolor=DARK_BG, edgecolor=GRID_C,
                    labelcolor=WHITE, fontsize=7)

    # ── Título y anotación central ──
    fig.suptitle(
        "GBM bajo distintos drifts  —  el mundo real $\\mathbb{P}$  vs  el mundo neutro al riesgo $\\mathbb{Q}$",
        color=WHITE, fontsize=14, y=0.99)

    # Flecha destacando la columna central
    fig.text(0.50, 0.94,
             "⬇  Bajo $\\mathbb{Q}$, usamos $\\mu = r$ para valorar derivados\n"
             "   (el drift real desaparece del precio del derivado)",
             ha="center", va="top", color=CYAN, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d2030",
                       edgecolor=CYAN, alpha=0.85))

    if save_fig:
        plt.savefig("04_gbm_drift_comparison.png", dpi=150,
                    bbox_inches="tight", facecolor=DARK_BG)
        print("Guardado: 04_gbm_drift_comparison.png")
    plt.show()

# ==========================================================================
# 6. Monte Carlo: Convergencia detallada + intervalo de confianza
# ==========================================================================
def montecarlo_convergence(save_fig=True):
    """
    4 paneles:
     (A) 100 trayectorias GBM bajo Q + strike
     (B) Distribución de S_T con zona ITM sombreada
     (C) Distribución de pagos descontados e^{-rT}Φ(S_T)
     (D) Convergencia del estimador MC con IC 95%
    """
    np.random.seed(42)
    S0, K, T_end, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    N_MC, N_plot, steps = 50_000, 100, 252
    dt  = T_end / steps
    t   = np.linspace(0, T_end, steps+1)

    def bs_call(S, K, T, r, sig):
        if T <= 0 or sig <= 0: return max(S-K, 0)*np.exp(-r*T)
        d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    # ── Trayectorias visibles ──
    paths = np.zeros((N_plot, steps+1))
    paths[:, 0] = S0
    for i in range(steps):
        Z = np.random.randn(N_plot)
        paths[:, i+1] = paths[:, i] * np.exp(
            (r-.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    # ── Monte Carlo grande ──
    Z_big  = np.random.randn(N_MC)
    S_T    = S0 * np.exp((r-.5*sigma**2)*T_end + sigma*np.sqrt(T_end)*Z_big)
    payoff = np.maximum(S_T - K, 0)
    disc_p = np.exp(-r*T_end) * payoff

    bs_price = bs_call(S0, K, T_end, r, sigma)
    cum_avg  = np.cumsum(disc_p) / (np.arange(N_MC) + 1)
    cum_std  = np.array([disc_p[:i+1].std(ddof=1)/np.sqrt(i+1)
                         for i in range(0, N_MC, 500)])
    n_x      = np.arange(1, N_MC+1, 500)

    # ── Figura ──
    fig = plt.figure(figsize=(14, 9))
    _style_dark(fig)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    [_style_dark(fig, ax) for ax in axes]

    # (A) Trayectorias
    ax = axes[0]
    cmap = plt.cm.cool
    for i in range(N_plot):
        ax.plot(t, paths[i], lw=0.6, alpha=0.28, color=cmap(i/N_plot))
    ax.axhline(K, color=ORANGE, lw=2, ls="--", label=f"Strike K = {K}")
    ax.plot(t, S0*np.exp(r*t), color=WHITE, lw=2, ls=":", label=r"$S_0 e^{rt}$ (media $\mathbb{Q}$)")
    ax.set_title(r"① Trayectorias GBM bajo $\mathbb{Q}$", fontsize=11)
    ax.set_xlabel("t"); ax.set_ylabel("$S_t$")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    # (B) Distribución S_T con zona ITM
    ax = axes[1]
    n_bins = 60
    counts, edges, patches = ax.hist(S_T, bins=n_bins, density=True,
                                     color=CYAN, alpha=0.5, edgecolor=DARK_BG)
    for patch, left in zip(patches, edges[:-1]):
        if left > K:
            patch.set_facecolor(GREEN)
            patch.set_alpha(0.75)
    # PDF log-normal teórico
    x_g = np.linspace(S_T.min(), S_T.max(), 400)
    lmu = np.log(S0) + (r-.5*sigma**2)*T_end
    lsig = sigma*np.sqrt(T_end)
    ax.plot(x_g, norm.pdf(np.log(x_g), lmu, lsig)/x_g,
            color=WHITE, lw=2, label="PDF log-normal teórica")
    ax.axvline(K, color=ORANGE, lw=2, ls="--", label=f"K = {K}")
    p_itm = (S_T > K).mean()
    ax.text(0.97, 0.97, f"P(ITM) = {p_itm:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            color=GREEN, fontsize=10,
            bbox=dict(fc=DARK_BG, ec=GREEN, pad=3))
    ax.set_title(r"② Distribución $S_T$  —  verde = ITM", fontsize=11)
    ax.set_xlabel("$S_T$"); ax.set_ylabel("Densidad")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    # (C) Pagos descontados
    ax = axes[2]
    ax.hist(disc_p, bins=60, density=True, color=PURPLE,
            alpha=0.6, edgecolor=DARK_BG)
    ax.axvline(disc_p.mean(), color=WHITE, lw=2.5, ls="--",
               label=f"MC = {disc_p.mean():.4f}")
    ax.axvline(bs_price, color=ORANGE, lw=2.5, ls=":",
               label=f"B–S = {bs_price:.4f}")
    ax.set_title(r"③ Pagos descontados  $e^{-rT}\,\Phi(S_T)$", fontsize=11)
    ax.set_xlabel("Valor descontado"); ax.set_ylabel("Densidad")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    # (D) Convergencia con IC
    ax = axes[3]
    ax.plot(np.arange(1, N_MC+1), cum_avg, color=CYAN, lw=1.2,
            label="Estimador MC")
    ax.fill_between(n_x,
                    cum_avg[n_x-1] - 1.96*cum_std,
                    cum_avg[n_x-1] + 1.96*cum_std,
                    alpha=0.25, color=CYAN, label="IC 95%")
    ax.axhline(bs_price, color=ORANGE, lw=2.5, ls="--",
               label=f"B–S = {bs_price:.4f}")
    ax.set_title("④ Convergencia MC  →  solución Feynman–Kač", fontsize=11)
    ax.set_xlabel("N simulaciones"); ax.set_ylabel("Precio estimado")
    ax.set_xscale("log")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    fig.suptitle(
        r"Monte Carlo  $\longleftrightarrow$  Feynman–Kač  —  "
        r"$f_t = e^{-r\tau}\,\mathbb{E}^{\mathbb{Q}}[f_T\,|\,S_t]$",
        color=WHITE, fontsize=14)

    print(f"\n=== Resumen ===")
    print(f"Precio Monte Carlo (N={N_MC:,}): {disc_p.mean():.6f}")
    print(f"Precio Black–Scholes analítico: {bs_price:.6f}")
    print(f"Error estándar MC:              {disc_p.std()/np.sqrt(N_MC):.6f}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_fig:
        plt.savefig("06_montecarlo.png", dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print("Guardado: 06_montecarlo.png")
    plt.show()


# ==========================================================================
# 7. full_story() — Toda la narrativa en un solo panel de clase
# ==========================================================================
def full_story(save_fig=True):
    """
    Panel único 2×3 que resume el arco completo del curso:
      [0,0] BM 1D
      [0,1] Difusión de partículas → distribución Normal
      [0,2] Tres drifts (P vs Q)
      [1,0] Trayectorias GBM bajo Q
      [1,1] Distribución S_T con zona ITM
      [1,2] Convergencia MC → precio B–S
    """
    np.random.seed(0)
    S0, K, r, sigma, T_end = 100.0, 100.0, 0.05, 0.20, 1.0
    dt_s = 1/252; steps = int(T_end/dt_s)
    t_s  = np.linspace(0, T_end, steps+1)
    dt_b = 0.01;  T_b = 300
    t_b  = np.linspace(0, T_b*dt_b, T_b)

    # — BM 1D —
    W1 = np.concatenate([[0], np.cumsum(np.sqrt(dt_b)*np.random.randn(T_b-1))])

    # — Difusión N=80 partículas —
    N_diff = 80
    dW_d = np.sqrt(dt_b)*np.random.randn(N_diff, T_b)
    W_d  = np.cumsum(dW_d, axis=1)

    # — GBM tres drifts —
    mus = [0.18, r, -0.08]
    clrs3 = [GREEN, CYAN, RED]
    N3, steps3 = 30, steps
    paths3 = {}
    for mu in mus:
        dW3 = np.sqrt(dt_s)*np.random.randn(N3, steps3)
        W3  = np.zeros((N3, steps3+1))
        W3[:,1:] = np.cumsum(dW3, axis=1)
        paths3[mu] = S0*np.exp((mu-.5*sigma**2)*t_s[None,:]+sigma*W3)

    # — GBM bajo Q —
    N_q = 80
    paths_q = np.zeros((N_q, steps+1)); paths_q[:,0] = S0
    for i in range(steps):
        Z = np.random.randn(N_q)
        paths_q[:,i+1] = paths_q[:,i]*np.exp((r-.5*sigma**2)*dt_s+sigma*np.sqrt(dt_s)*Z)

    # — MC —
    N_MC = 30_000
    S_T_mc = S0*np.exp((r-.5*sigma**2)*T_end+sigma*np.sqrt(T_end)*np.random.randn(N_MC))
    payoff_mc = np.maximum(S_T_mc-K,0)
    disc_mc   = np.exp(-r*T_end)*payoff_mc
    cum_avg   = np.cumsum(disc_mc)/(np.arange(N_MC)+1)
    d1 = (np.log(S0/K)+(r+.5*sigma**2)*T_end)/(sigma*np.sqrt(T_end))
    bs_v = S0*norm.cdf(d1) - K*np.exp(-r*T_end)*norm.cdf(d1-sigma*np.sqrt(T_end))

    # ── Figura ──
    fig = plt.figure(figsize=(18, 10))
    _style_dark(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)
    axes = [[fig.add_subplot(gs[r,c]) for c in range(3)] for r in range(2)]
    [_style_dark(fig, axes[r][c]) for r in range(2) for c in range(3)]

    # [0,0] BM 1D
    ax = axes[0][0]
    n  = len(t_b)
    alphas_line = np.linspace(0.2, 1.0, n-1)
    rgba = np.zeros((n-1,4))
    rgba[:,0]=0.34; rgba[:,1]=0.65; rgba[:,2]=1.0; rgba[:,3]=alphas_line
    segs = np.stack([np.column_stack([t_b[:-1],W1[:-1]]),
                     np.column_stack([t_b[1:], W1[1:]])], axis=1)
    lc = LineCollection(segs, colors=rgba, linewidths=1.6)
    ax.add_collection(lc)
    band_t = np.linspace(0.001, t_b[-1], 200)
    ax.fill_between(band_t, -np.sqrt(band_t), np.sqrt(band_t),
                    alpha=0.15, color=CYAN)
    ax.set_xlim(t_b[0], t_b[-1]); ax.set_ylim(W1.min()-0.5, W1.max()+0.5)
    ax.set_title("① BM — Una partícula  $W(t)$", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("W(t)")

    # [0,1] Difusión N partículas
    ax = axes[0][1]
    cm = plt.cm.cool
    for i in range(N_diff):
        ax.plot(t_b, W_d[i], lw=0.5, alpha=0.3, color=cm(i/N_diff))
    ax.fill_between(t_b, -np.sqrt(t_b), np.sqrt(t_b),
                    alpha=0.15, color=CYAN, label=r"$\pm\sqrt{t}$")
    ax.set_title(f"② Difusión  ({N_diff} partículas)", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("W(t)")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=8)

    # [0,2] Tres drifts
    ax = axes[0][2]
    mu_labels = [r"$\mu=0.18$ (P alcista)", r"$\mu=r=0.05$ (Q)", r"$\mu=-0.08$ (P bajista)"]
    for mu, col, lbl in zip(mus, clrs3, mu_labels):
        for i in range(N3):
            ax.plot(t_s, paths3[mu][i], lw=0.6, alpha=0.25, color=col)
        ax.plot(t_s, S0*np.exp(mu*t_s), color=col, lw=2.2, ls="--", label=lbl)
    ax.axhline(S0, color=GRID_C, lw=0.8, ls=":")
    ax.set_title(r"③ GBM — $\mathbb{P}$ vs $\mathbb{Q}$", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("$S_t$")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=7.5)

    # [1,0] Trayectorias bajo Q
    ax = axes[1][0]
    cm2 = plt.cm.plasma
    for i in range(N_q):
        ax.plot(t_s, paths_q[i], lw=0.6, alpha=0.25, color=cm2(i/N_q))
    ax.axhline(K, color=ORANGE, lw=2, ls="--", label=f"K = {K}")
    ax.plot(t_s, S0*np.exp(r*t_s), color=WHITE, lw=2, ls=":",
            label=r"$S_0 e^{rt}$")
    ax.set_title(r"④ Trayectorias GBM bajo $\mathbb{Q}$", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("$S_t$")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=8)

    # [1,1] Distribución S_T
    ax = axes[1][1]
    cnts, edgs, ptchs = ax.hist(S_T_mc, bins=50, density=True,
                                color=CYAN, alpha=0.45, edgecolor=DARK_BG)
    for patch, left in zip(ptchs, edgs[:-1]):
        if left > K:
            patch.set_facecolor(GREEN); patch.set_alpha(0.75)
    ax.axvline(K, color=ORANGE, lw=2, ls="--", label=f"K={K}")
    p_itm = (S_T_mc>K).mean()
    ax.text(0.97, 0.95, f"P(ITM) ≈ {p_itm:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            color=GREEN, fontsize=9,
            bbox=dict(fc=DARK_BG, ec=GREEN, pad=3))
    ax.set_title(r"⑤ Distribución $S_T$  — verde = ITM", fontsize=10)
    ax.set_xlabel("$S_T$"); ax.set_ylabel("Densidad")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=8)

    # [1,2] Convergencia MC
    ax = axes[1][2]
    ax.plot(np.arange(1,N_MC+1), cum_avg, color=CYAN, lw=1.3,
            label="Estimador MC")
    ax.axhline(bs_v, color=ORANGE, lw=2.5, ls="--",
               label=f"B–S = {bs_v:.4f}")
    ax.fill_between(np.arange(1,N_MC+1),
                    cum_avg - 1.96*disc_mc.std()/np.sqrt(np.arange(1,N_MC+1)),
                    cum_avg + 1.96*disc_mc.std()/np.sqrt(np.arange(1,N_MC+1)),
                    alpha=0.2, color=CYAN)
    ax.set_xscale("log")
    ax.set_title(r"⑥ MC converge a Feynman–Kač  →  B–S", fontsize=10)
    ax.set_xlabel("N"); ax.set_ylabel("Precio")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_C, labelcolor=WHITE, fontsize=9)

    fig.suptitle(
        "Del Movimiento Browniano a Feynman–Kač  —  Arco completo del curso",
        color=WHITE, fontsize=15, y=0.99)

    plt.tight_layout(rect=[0,0,1,0.97])
    if save_fig:
        plt.savefig("07_full_story.png", dpi=160, bbox_inches="tight",
                    facecolor=DARK_BG)
        print("Guardado: 07_full_story.png")
    plt.show()


# ==========================================================================
# Punto de entrada
# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Quantitative Finance — ITESO")
    print("  Brownian Motion → GBM → Feynman–Kač → Black–Scholes")
    print("=" * 60)

    print("\n[1/6] BM — una sola partícula (animación)…")
    bm_single_particle()
    # ── Módulos estáticos (figuras) ──────────────────────────
    print("\n[2/6] Difusión de partículas…")
    bm_diffusion()

    print("\n[3/6] BM 1D → precio de acción…")
    bm_1d_to_stock()

    print("\n[4/6] Comparación de drifts (P vs Q)…")
    gbm_drift_comparison()

    print("\n[5/6] Monte Carlo — convergencia…")
    montecarlo_convergence()

    print("\n[6/6] Panel completo (full story)…")
    full_story()

    # ── Módulos animados (últimos, requieren ventana interactiva) ─
    
