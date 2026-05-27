"""
Visualización interactiva de Maximum Likelihood Estimation (MLE)
---------------------------------------------------------------
Dependencias: numpy, matplotlib  (tkinter viene con Python)
Instalar:     pip install numpy matplotlib

Correr:       python mle_interactivo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

RNG = np.random.default_rng()
SIGMA = 1.5
N = 6
X_RANGE = np.linspace(-1, 12, 400)
MU_RANGE = np.linspace(0, 10, 400)


# ── funciones ────────────────────────────────────────────────────────────────

def normal_pdf(x, mu, sigma=SIGMA):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def log_likelihood(mu, data):
    return np.sum(np.log(normal_pdf(data, mu)))


def generar_datos():
    true_mu = 4 + RNG.random() * 3
    return RNG.normal(loc=true_mu, scale=SIGMA, size=N)


# ── figura ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 8), facecolor="#fafafa")
fig.canvas.manager.set_window_title("MLE interactivo — μ candidato")

gs = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[2.2, 2, 0.7],
    hspace=0.45,
    top=0.93, bottom=0.08, left=0.10, right=0.95
)

ax_dens = fig.add_subplot(gs[0])
ax_ll   = fig.add_subplot(gs[1])
ax_ctrl = fig.add_subplot(gs[2])
ax_ctrl.axis("off")

BLUE   = "#185FA5"
GREEN  = "#0F6E56"
AMBER  = "#BA7517"
GRAY   = "#73726c"
LIGHT  = "#E6F1FB"

# ── estado inicial ────────────────────────────────────────────────────────────

data = generar_datos()
mu_init = 5.0


# ── gráfica superior: densidad ────────────────────────────────────────────────

ax_dens.set_title("Densidad f(x | μ) y datos observados", fontsize=12, pad=8, color="#2C2C2A")
ax_dens.set_xlabel("x", fontsize=11, color=GRAY)
ax_dens.set_ylabel("f(x | μ)", fontsize=11, color=GRAY)
ax_dens.set_xlim(-1, 12)
ax_dens.tick_params(colors=GRAY, labelsize=9)
for spine in ax_dens.spines.values():
    spine.set_edgecolor("#D3D1C7")
ax_dens.set_facecolor("white")
ax_dens.grid(True, color="#D3D1C7", linewidth=0.5, linestyle="--", alpha=0.6)

curve_line, = ax_dens.plot(X_RANGE, normal_pdf(X_RANGE, mu_init), color=BLUE, lw=2, label="f(x|μ)")
fill_poly   = ax_dens.fill_between(X_RANGE, normal_pdf(X_RANGE, mu_init), alpha=0.08, color=BLUE)

vline_mu  = ax_dens.axvline(mu_init, color=BLUE, lw=1.5, ls="--", label="μ candidato")
vline_mle = ax_dens.axvline(np.mean(data), color=GREEN, lw=1.5, ls="-", label="x̄ (MLE)")

stem_lines  = []
stem_points = []

def draw_stems(mu_val):
    for ln in stem_lines:
        ln.remove()
    for pt in stem_points:
        pt.remove()
    stem_lines.clear()
    stem_points.clear()
    for xi in data:
        h = normal_pdf(xi, mu_val)
        ln, = ax_dens.plot([xi, xi], [0, h], color=AMBER, lw=1.2, ls=":", alpha=0.7)
        pt, = ax_dens.plot(xi, 0, "o", color=AMBER, ms=6, zorder=5)
        stem_lines.append(ln)
        stem_points.append(pt)

draw_stems(mu_init)

legend = ax_dens.legend(fontsize=9, loc="upper right", framealpha=0.9,
                        edgecolor="#D3D1C7", facecolor="white")

# cuadro de stats arriba
stats_text = ax_dens.text(
    0.02, 0.97, "", transform=ax_dens.transAxes,
    fontsize=9, va="top", ha="left", color="#2C2C2A",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D3D1C7", alpha=0.9)
)


# ── gráfica inferior: log-likelihood ─────────────────────────────────────────

ax_ll.set_title("Log-likelihood  ℓ(μ)", fontsize=12, pad=8, color="#2C2C2A")
ax_ll.set_xlabel("μ candidato", fontsize=11, color=GRAY)
ax_ll.set_ylabel("ℓ(μ)", fontsize=11, color=GRAY)
ax_ll.set_xlim(0, 10)
ax_ll.tick_params(colors=GRAY, labelsize=9)
for spine in ax_ll.spines.values():
    spine.set_edgecolor("#D3D1C7")
ax_ll.set_facecolor("white")
ax_ll.grid(True, color="#D3D1C7", linewidth=0.5, linestyle="--", alpha=0.6)

ll_vals  = np.array([log_likelihood(m, data) for m in MU_RANGE])
ll_line, = ax_ll.plot(MU_RANGE, ll_vals, color=GREEN, lw=2)

ll_point, = ax_ll.plot(mu_init, log_likelihood(mu_init, data), "o",
                       color=AMBER, ms=9, zorder=6, label="μ actual")
ll_max_pt, = ax_ll.plot(np.mean(data), log_likelihood(np.mean(data), data), "^",
                        color=GREEN, ms=9, zorder=6, label="máximo (x̄)")
ax_ll.legend(fontsize=9, loc="lower right", framealpha=0.9,
             edgecolor="#D3D1C7", facecolor="white")

ann_max = ax_ll.annotate(
    "↑ máximo", xy=(np.mean(data), log_likelihood(np.mean(data), data)),
    xytext=(0, 14), textcoords="offset points",
    ha="center", fontsize=8, color=GREEN
)


# ── controles ────────────────────────────────────────────────────────────────

ax_slider = plt.axes([0.12, 0.035, 0.58, 0.025])
slider_mu = Slider(ax_slider, "μ", 0.0, 10.0, valinit=mu_init, valstep=0.05,
                   color=BLUE, track_color="#D3D1C7")
slider_mu.label.set_fontsize(10)
slider_mu.valtext.set_fontsize(10)

ax_btn_resample = plt.axes([0.75, 0.025, 0.10, 0.040])
ax_btn_mle      = plt.axes([0.86, 0.025, 0.09, 0.040])
btn_resample = Button(ax_btn_resample, "Nuevos datos", color="#F1EFE8", hovercolor="#D3D1C7")
btn_mle      = Button(ax_btn_mle,      "Ir al MLE",    color="#E6F1FB", hovercolor="#B5D4F4")
btn_resample.label.set_fontsize(9)
btn_mle.label.set_fontsize(9)


# ── función de actualización ──────────────────────────────────────────────────

def update(mu_val=None):
    if mu_val is None:
        mu_val = slider_mu.val

    xbar = np.mean(data)

    # curva densidad
    y_curve = normal_pdf(X_RANGE, mu_val)
    curve_line.set_ydata(y_curve)

    # rellenar (remake)
    global fill_poly
    fill_poly.remove()
    fill_poly = ax_dens.fill_between(X_RANGE, y_curve, alpha=0.08, color=BLUE)

    vline_mu.set_xdata([mu_val, mu_val])
    vline_mle.set_xdata([xbar, xbar])
    draw_stems(mu_val)
    ax_dens.relim()
    ax_dens.autoscale_view(scalex=False)

    # log-likelihood
    ll_val = log_likelihood(mu_val, data)
    ll_mle = log_likelihood(xbar, data)
    ll_point.set_data([mu_val], [ll_val])
    ll_max_pt.set_data([xbar], [ll_mle])
    ann_max.xy = (xbar, ll_mle)
    ann_max.set_position((0, 14))

    # stats box
    l_val = np.prod(normal_pdf(data, mu_val))
    stats_text.set_text(
        f"μ candidato = {mu_val:.2f}   |   x̄ (MLE) = {xbar:.2f}\n"
        f"log-likelihood = {ll_val:.2f}   |   likelihood = {l_val:.2e}"
    )

    fig.canvas.draw_idle()


def on_slider(val):
    update(val)

def on_resample(event):
    global data
    data = generar_datos()
    xbar = np.mean(data)
    # recalcular curva ll
    new_ll = np.array([log_likelihood(m, data) for m in MU_RANGE])
    ll_line.set_ydata(new_ll)
    ax_ll.relim()
    ax_ll.autoscale_view(scalex=False)
    ann_max.xy = (xbar, log_likelihood(xbar, data))
    update(slider_mu.val)

def on_goto_mle(event):
    xbar = float(np.mean(data))
    xbar_clipped = np.clip(xbar, 0, 10)
    slider_mu.set_val(round(xbar_clipped, 2))
    update(xbar_clipped)

slider_mu.on_changed(on_slider)
btn_resample.on_clicked(on_resample)
btn_mle.on_clicked(on_goto_mle)

update(mu_init)
plt.show()
