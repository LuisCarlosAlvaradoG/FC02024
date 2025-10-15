import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.stats import norm

# Animación Browniana 2D estilo “neón” sobre fondo negro --------------------------------------------------------------------------------------------------------------------------------
np.random.seed(7)     # para reproducibilidad
N = 120               # número de partículas
T = 200               # frames totales
dt = 0.02             # paso temporal
sigma = 0.35          # difusión (más grande -> se dispersa más)
cluster_sigma = 0.03  # cuán apretado inicia el grupo
trail_layers = 4      # capas para la estela (0 = sin estela)

# =============== Estado inicial ==============
x = np.random.normal(0.0, cluster_sigma, size=N)
y = np.random.normal(0.0, cluster_sigma, size=N)
traj_x = np.empty((T, N))
traj_y = np.empty((T, N))
traj_x[0] = x
traj_y[0] = y

# Increments brownianos: N(0, dt) en cada eje, independientes
dW_x = np.random.normal(0.0, np.sqrt(dt), size=(T-1, N))
dW_y = np.random.normal(0.0, np.sqrt(dt), size=(T-1, N))

for t in range(1, T):
    x = x + sigma * dW_x[t-1]
    y = y + sigma * dW_y[t-1]
    traj_x[t] = x
    traj_y[t] = y

# =============== Figura/Estilo ===============
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# Colores fijos por partícula (paleta tipo “neón”)
colors = plt.cm.hsv(np.linspace(0, 1, N, endpoint=False))
np.random.shuffle(colors)

# Capas para estela: tamaños y alphas decrecientes
sizes = [26, 18, 12, 8][:trail_layers]
alphas = [1.0, 0.55, 0.28, 0.14][:trail_layers]
scatters = []
for k in range(trail_layers):
    sc = ax.scatter(traj_x[0], traj_y[0], s=sizes[k], c=colors, alpha=alphas[k], linewidths=0)
    scatters.append(sc)

title_txt = ax.text(0.02, 0.96, "Browniano 2D — Dispersión de 'polen'",
                    transform=ax.transAxes, color="white", fontsize=10, ha="left", va="top")
time_txt = ax.text(0.98, 0.96, "t = 0",
                   transform=ax.transAxes, color="white", fontsize=10, ha="right", va="top")

def init():
    for k in range(trail_layers):
        scatters[k].set_offsets(np.column_stack([traj_x[0], traj_y[0]]))
    time_txt.set_text("t = 0")
    return (*scatters, time_txt, title_txt)

def update(frame):
    # Capa 0 = estado actual; capas siguientes = frames anteriores
    for k in range(trail_layers):
        idx = max(0, frame - k)
        scatters[k].set_offsets(np.column_stack([traj_x[idx], traj_y[idx]]))
    time_txt.set_text(f"t = {frame}")
    return (*scatters, time_txt)

anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=35)

# =============== Guardar como GIF ===============
# Requiere Pillow instalado: pip install pillow
gif_path = "brownian_black_neon.gif"
anim.save(gif_path, writer=PillowWriter(fps=24))
print(f"GIF guardado en: {gif_path}")

# Opcional: mostrar en una ventana (si corres localmente)
plt.show()

# Browniano 2D de una partícula con estela que se desvanece --------------------------------------------------------------------------------------------------------------------------------
np.random.seed(1)
T = 600          # frames totales
dt = 0.02        # tamaño de paso
sigma = 0.55     # difusión (mayor -> pasos más grandes)
trail_len = 150  # longitud de estela (segmentos recientes que se muestran)

# ---------------- Trayectoria ----------------
x = np.zeros(T)
y = np.zeros(T)
# incrementos ~ N(0, sigma^2 dt)
dx = sigma * np.sqrt(dt) * np.random.randn(T-1)
dy = sigma * np.sqrt(dt) * np.random.randn(T-1)
x[1:] = np.cumsum(dx)
y[1:] = np.cumsum(dy)

# ---------------- Utilidad ----------------
def make_segments(xv, yv):
    """Convierte puntos en segmentos Nx2x2 para LineCollection."""
    pts = np.column_stack([xv, yv])
    return np.stack([pts[:-1], pts[1:]], axis=1)

# ---------------- Figura ----------------
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(x.min()-2, x.max()+2)
ax.set_ylim(y.min()-2, y.max()+2)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# LineCollection para la estela
segments = make_segments(x[:2], y[:2])
lc = LineCollection(segments, linewidths=2)
ax.add_collection(lc)

# Punto actual (cabeza)
head, = ax.plot([x[0]], [y[0]], "o", ms=8, color="white")

# Texto de tiempo
time_txt = ax.text(0.98, 0.96, "t = 0", transform=ax.transAxes,
                   color="white", ha="right", va="top", fontsize=10)

# Colores base para la línea (tono azulado/verdoso) con alpha creciente
base_rgb = np.array([0.0, 0.9, 0.8])  # ajusta a gusto

def update(frame):
    # Índices para los últimos trail_len puntos
    start = max(0, frame - trail_len)
    xv = x[start:frame+1]
    yv = y[start:frame+1]

    if len(xv) >= 2:
        segs = make_segments(xv, yv)
        lc.set_segments(segs)

        # alpha de 0.05 -> 1.0 en función de la antigüedad (fade)
        nseg = len(segs)
        alphas = np.linspace(0.05, 1.0, nseg)
        colors = np.tile(base_rgb, (nseg, 1))
        colors = np.concatenate([colors, alphas[:, None]], axis=1)  # RGBA
        lc.set_color(colors)

    # Actualiza cabeza
    head.set_data([x[frame]], [y[frame]])
    time_txt.set_text(f"t = {frame}")
    return lc, head, time_txt

anim = FuncAnimation(fig, update, frames=T, interval=20, blit=True)

# ---- Guardar a GIF (opcional) ----
# anim.save("browniano_unaparte_fade.gif", writer=PillowWriter(fps=30))

# Mostrar (si corres localmente)
plt.show()

# 1D Movimiento Browniano ----------------------------------------------------------------------------------------------------------------------------------------------------------------
np.random.seed(0)
T   = 300      # frames
dt  = 0.01
mu  = 0.0
sig = 1.0

# incrementos y trayectoria
dW = np.random.normal(loc=mu*dt, scale=np.sqrt(dt), size=T)
W  = np.cumsum(dW)
t  = np.linspace(0, T*dt, T)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Movimiento Browniano 1D — Animación")
ax.set_xlabel("Tiempo (t)")
ax.set_ylabel("W(t)")
ax.grid(alpha=0.3)

line,  = ax.plot([], [], lw=2)
point, = ax.plot([], [], "o")
time_txt = ax.text(0.98, 0.9, "", transform=ax.transAxes, ha="right")

m = 0.15 * max(1.0, np.max(np.abs(W)))
ax.set_xlim(t[0], t[-1])
ax.set_ylim(W.min()-m, W.max()+m)

def init():
    line.set_data([], [])
    point.set_data([], [])
    time_txt.set_text("")
    return line, point, time_txt

def update(frame):
    line.set_data(t[:frame+1], W[:frame+1])
    point.set_data(t[frame], W[frame])
    time_txt.set_text(f"t = {t[frame]:.2f}")
    return line, point, time_txt

ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=20)

# Mostrar en tiempo real
plt.show()


# Animación: múltiples trayectorias brownianas 1D + histograma de posiciones finales --------------------------------------------------------------------------------------------------------------------------------
np.random.seed(42)
N   = 120          # número de partículas (sube/baja según tu máquina)
T   = 500          # pasos de tiempo
dt  = 0.01
mu  = 0.0          # drift del browniano puro
sigma = 1.0        # escala del ruido

# Bins para el histograma final (comparte escala vertical del panel izquierdo)
num_bins = 25

# ---------------- Simulación (precomputada para animar ágil) ----------------
# dW ~ N(mu*dt, sigma*sqrt(dt))
dW = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt), size=(N, T))
W  = np.cumsum(dW, axis=1)   # shape: (N, T)
t  = np.linspace(0, T*dt, T) # tiempo continuo

# Posiciones finales (para el histograma que se irá “revelando”)
final_positions = W[:, -1].copy()

# ---------------- Figura y ejes ----------------
plt.style.use("default")
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.0, 1.0], wspace=0.25)

ax_traj = fig.add_subplot(gs[0, 0])
ax_hist = fig.add_subplot(gs[0, 1])

ax_traj.set_title("Evolving Trajectories")
ax_traj.set_xlabel("Time (t)")
ax_traj.set_ylabel("Position")
ax_traj.grid(alpha=0.25)

# Fijamos límites para que no "salten" durante la animación
pad_y = 0.15 * np.max(np.abs(W))
ax_traj.set_xlim(t[0], t[-1])
ax_traj.set_ylim(W.min() - pad_y, W.max() + pad_y)

# Creamos las líneas de cada trayectoria (alpha bajo para no saturar)
lines = []
for i in range(N):
    ln, = ax_traj.plot([], [], lw=1.0, alpha=0.6)
    lines.append(ln)

# Línea vertical que marca el tiempo actual (opcional)
time_marker = ax_traj.axvline(t[0], lw=1, alpha=0.5)

# ---------- Histograma horizontal (panel derecho) ----------
ax_hist.set_title("Final Position Histogram")
ax_hist.set_xlabel("Count")
# Eje Y compartido con el rango del panel izquierdo para que coincidan escalas visuales
ymin, ymax = ax_traj.get_ylim()
bins = np.linspace(ymin, ymax, num_bins+1)
centers = 0.5*(bins[1:] + bins[:-1])

# Inicializamos con ceros (sin “llegadas” aún)
counts, _ = np.histogram([], bins=bins)
bars = ax_hist.barh(centers, counts, height=(bins[1]-bins[0])*0.9)

# Escala X del histograma: estimación tosca del máximo posible
ax_hist.set_xlim(0, max(1, int(N*0.25)))  # se ajustará en la animación si hace falta
ax_hist.grid(alpha=0.25, axis="x")
ax_hist.set_ylim(ymin, ymax)

# Texto de progreso
progress_txt = ax_hist.text(0.98, 0.02, "", transform=ax_hist.transAxes,
                             ha="right", va="bottom")

# ---------------- Funciones de animación ----------------
def init():
    # Trayectorias vacías al inicio
    for ln in lines:
        ln.set_data([], [])
    time_marker.set_xdata(t[0])

    # Histograma a cero
    for rect in bars:
        rect.set_width(0.0)
    progress_txt.set_text("0 / {} finalized".format(N))

    return (*lines, time_marker, *bars, progress_txt)

def update(frame):
    # 1) Actualiza trayectorias hasta el frame actual
    #    (para eficiencia no “re-dibuja” todo; sólo corta hasta frame)
    xdata = t[:frame+1]
    for i, ln in enumerate(lines):
        ydata = W[i, :frame+1]
        ln.set_data(xdata, ydata)

    time_marker.set_xdata(t[frame])

    # 2) Vamos “revelando” gradualmente las posiciones finales:
    #    al frame f, contamos a los primeros k = floor((f/T)*N)
    k = int((frame + 1) / T * N)  # cuántas finales ya “llegaron”
    revealed = final_positions[:k]
    counts, _ = np.histogram(revealed, bins=bins)

    # Ajusta ancho de barras
    max_count = counts.max() if counts.size > 0 else 1
    if max_count > ax_hist.get_xlim()[1]:
        ax_hist.set_xlim(0, max_count * 1.1)

    for rect, c in zip(bars, counts):
        rect.set_width(c)

    progress_txt.set_text(f"{k} / {N} finalized")

    return (*lines, time_marker, *bars, progress_txt)

# ---------------- Ejecutar animación ----------------
anim = FuncAnimation(
    fig, update, frames=T, init_func=init,
    interval=20, blit=False  # blit=False por la cantidad de artistas
)

# Muestra en vivo (si corres localmente)
plt.show()

# HEATMAP ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# Trayectorias brownianas 1D + Heatmap (tiempo–posición) generados SIMULTÁNEAMENTE
# - Simulación en línea (frame a frame)
# - Ventana interactiva con plt.show()
# - (Opcional) guarda GIF y PNG del heatmap final

np.random.seed(42)
N   = 120        # nº de partículas
T   = 300        # nº de pasos de tiempo (frames)
dt  = 0.01       # paso temporal
mu  = 0.0        # drift
sigma = 1.0      # ruido
pos_bins   = 70  # resolución vertical del heatmap
use_log    = True
interval_ms = 25 # velocidad de animación (ms/frame)

# Guardado (opcional)
SAVE_GIF = False               # pon True si quieres guardar el GIF
GIF_PATH = "brownian_heatmap_anim.gif"
PNG_PATH = "brownian_heatmap_final.png"

# ===================== Preparación simulación =====================
t = np.linspace(0, (T-1)*dt, T)            # eje de tiempo
dW = np.random.normal(mu*dt, sigma*np.sqrt(dt), size=(N, T))  # incrementos pre-generados

# Estado que se actualizará en vivo
pos   = np.zeros(N)                         # posiciones actuales
paths = np.full((N, T), np.nan, dtype=float)  # para dibujar trayectorias
paths[:, 0] = pos

# Rango vertical (se estima con una cota típica √t * sigma, y luego se expande si hace falta)
y_est = 3.5 * sigma * np.sqrt(T*dt)
ymin, ymax = -y_est, y_est
y_edges = np.linspace(ymin, ymax, pos_bins + 1)

# Matriz del heatmap que se irá llenando columna a columna
H = np.zeros((pos_bins, T), dtype=np.int32)

# ===================== Figura y ejes =====================
plt.style.use("default")
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.0, 1.8], wspace=0.25)

# Panel izquierdo: trayectorias (tiempo vs posición)
ax_traj = fig.add_subplot(gs[0, 0])
ax_traj.set_title("Evolving Brownian Trajectories (built online)")
ax_traj.set_xlabel("Time (t)")
ax_traj.set_ylabel("Position")
ax_traj.grid(alpha=0.25)
ax_traj.set_xlim(t[0], t[-1])
ax_traj.set_ylim(ymin, ymax)

# Una línea por partícula (inicialmente vacías)
lines = []
for _ in range(N):
    ln, = ax_traj.plot([], [], lw=1.0, alpha=0.55)
    lines.append(ln)

# Marcador de tiempo actual
time_marker = ax_traj.axvline(t[0], lw=1, alpha=0.6)

# Panel derecho: heatmap tiempo–posición (se construye a la par)
ax_heat = fig.add_subplot(gs[0, 1])
ax_heat.set_title("Occupancy Heatmap (Time vs Position)")
ax_heat.set_xlabel("Time (t)")
ax_heat.set_ylabel("Position")
ax_heat.set_xlim(t[0], t[-1])
ax_heat.set_ylim(ymin, ymax)

Z0 = np.zeros_like(H, dtype=float)
Z0 = np.log1p(Z0) if use_log else (Z0 / max(1, N))
im = ax_heat.imshow(
    Z0, origin="lower", aspect="auto",
    extent=[t[0], t[-1], y_edges[0], y_edges[-1]]
)
cbar = plt.colorbar(im, ax=ax_heat)
cbar.set_label("log(1 + count)" if use_log else "density")
time_txt = ax_heat.text(0.98, 0.02, "", transform=ax_heat.transAxes,
                        ha="right", va="bottom")

# ===================== Funciones de animación =====================
def init():
    # Deja todo vacío al inicio
    for ln in lines:
        ln.set_data([], [])
    time_marker.set_xdata(t[0])
    im.set_data(Z0)
    time_txt.set_text("")
    return (*lines, time_marker, im, time_txt)

def update(frame):
    global pos, paths, H

    # --- 1) Avanzar una iteración de la simulación ---
    if frame > 0:
        pos = pos + dW[:, frame]              # actualización de posiciones
        paths[:, frame] = pos                 # guardamos posiciones para dibujar

    # --- 2) Dibujar trayectorias hasta el frame actual ---
    xdata = t[:frame+1]
    for i, ln in enumerate(lines):
        ydata = paths[i, :frame+1]
        ln.set_data(xdata, ydata)

    time_marker.set_xdata(t[frame])

    # --- 3) Actualizar simultáneamente el heatmap en la columna actual ---
    # (solo la columna actual se rellena; así se "construye" a la par del tiempo)
    idx = np.digitize(pos, y_edges) - 1
    idx = np.clip(idx, 0, pos_bins - 1)
    H[:, frame] = np.bincount(idx, minlength=pos_bins)

    Z = np.log1p(H) if use_log else (H / max(1, N))
    im.set_data(Z)

    time_txt.set_text(f"t = {t[frame]:.2f}")
    return (*lines, time_marker, im, time_txt)

ani = FuncAnimation(fig, update, frames=T, init_func=init,
                    interval=interval_ms, blit=False)

if SAVE_GIF:
    try:
        ani.save(GIF_PATH, writer=PillowWriter(fps=max(1, int(1000/interval_ms))))
        print(f"GIF guardado en: {GIF_PATH}")
    except Exception as e:
        print("No se pudo guardar el GIF:", e)

def on_close(evt):
    # Guarda el heatmap final al cerrar la ventana
    Z_final = np.log1p(H) if use_log else (H / max(1, N))
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    im2 = ax2.imshow(
        Z_final, origin="lower", aspect="auto",
        extent=[t[0], t[-1], y_edges[0], y_edges[-1]]
    )
    ax2.set_title("Brownian Occupancy Heatmap — Final")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Position")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("log(1 + count)" if use_log else "density")
    plt.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig2)
    print(f"PNG del heatmap final guardado en: {PNG_PATH}")

fig.canvas.mpl_connect('close_event', on_close)

plt.show()

# Visualiza caminos GBM, la convergencia del precio MC y compara contra la fórmula cerrada.------------------------------------------------------------------------------------------
np.random.seed(42)
S0    = 100.0     # Precio inicial
K     = 100.0     # Strike
T     = 1.0       # Años a maduración
r     = 0.05      # Tasa libre de riesgo (constante)
sigma = 0.20      # Volatilidad anual
N_MC  = 100_000   # N simulaciones para el precio via Feynman-Kac
N_paths_plot = 100 # N trayectorias para visualizar
N_steps = 252     # Pasos por año para visualizar trayectorias

# ---------------------------
# Funciones auxiliares
# ---------------------------
def black_scholes_call(S, K, T, r, sigma):
    """Precio analítico de una call europea (Black–Scholes)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) * np.exp(-r*T)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def payoff_call(ST, K):
    return np.maximum(ST - K, 0.0)

# ---------------------------
# (1) Simulación de trayectorias GBM (para visualizar)
#     dS_t = r S_t dt + sigma S_t dW_t  (medida neutral al riesgo)
# ---------------------------
dt = T / N_steps
tgrid = np.linspace(0, T, N_steps + 1)

paths = np.empty((N_paths_plot, N_steps + 1))
paths[:, 0] = S0
for i in range(N_steps):
    Z = np.random.randn(N_paths_plot)
    # Solución exacta del incremento GBM en un paso
    paths[:, i+1] = paths[:, i] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

# ---------------------------
# (2) Simulación directa de S_T (vectorizada) para Monte Carlo
#     Usamos la solución cerrada: S_T = S0 * exp((r - 0.5*sigma^2)T + sigma sqrt(T) Z)
# ---------------------------
Z = np.random.randn(N_MC)
S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

payoffs = payoff_call(S_T, K)            # pagos a T
disc_payoffs = np.exp(-r*T) * payoffs    # (3) descuento a t

mc_price = disc_payoffs.mean()
mc_std   = disc_payoffs.std(ddof=1)
mc_se    = mc_std / np.sqrt(N_MC)        # error estándar

bs_price = black_scholes_call(S0, K, T, r, sigma)

# ---------------------------
# (4) Curva de convergencia del estimador Monte Carlo
# ---------------------------
# Para ver cómo converge el promedio al aumentar N
cum_avg = np.cumsum(disc_payoffs) / (np.arange(N_MC) + 1)

# ---------------------------
# Gráficos (4 subplots)
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# (1) Trayectorias GBM
ax = axs[0,0]
ax.plot(tgrid, paths.T, alpha=0.8)
ax.set_title("(1) Trayectorias GBM bajo Q")
ax.set_xlabel("Tiempo")
ax.set_ylabel("S_t")
ax.grid(True, alpha=0.3)

# (2) Pago a vencimiento Φ(S_T) (histograma + línea de strike)
ax = axs[0,1]
ax.hist(payoffs, bins=60, density=True, alpha=0.75)
# ax.axvline(K, color='k', linestyle='--', linewidth=1, alpha=0.7, label="Strike K (referencia)")
ax.set_title("(2) Pago a T: Φ(S_T)=max(S_T-K,0)")
ax.set_xlabel("Pago")
ax.set_ylabel("Densidad")
ax.legend()
ax.grid(True, alpha=0.3)

# (3) Pagos descontados e^{-rT} Φ(S_T)
ax = axs[1,0]
ax.hist(disc_payoffs, bins=60, density=True, alpha=0.75)
ax.set_title("(3) Pagos descontados a t: e^{-rT} Φ(S_T)")
ax.set_xlabel("Valor descontado")
ax.set_ylabel("Densidad")
ax.grid(True, alpha=0.3)

# (4) Convergencia del estimador MC al precio
ax = axs[1,1]
ax.plot(cum_avg, label="Estimador MC (promedio acumulado)")
ax.axhline(bs_price, color='k', linestyle='--', linewidth=1.5, label=f"Black–Scholes = {bs_price:.4f}")
ax.set_title("(4) Convergencia Monte Carlo → solución (Feynman–Kac)")
ax.set_xlabel("Número de simulaciones")
ax.set_ylabel("Precio estimado")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------
# Resumen en consola
# ---------------------------
print("=== Resumen Feynman–Kac (Call europea) ===")
print(f"Precio Monte Carlo (N={N_MC:,.0f}): {mc_price:.6f}  ± 1.96·SE ≈ [{mc_price-1.96*mc_se:.6f}, {mc_price+1.96*mc_se:.6f}]")
print(f"Precio Black–Scholes (analítico):     {bs_price:.6f}")
print(f"Error estándar MC:                     {mc_se:.6f}")
