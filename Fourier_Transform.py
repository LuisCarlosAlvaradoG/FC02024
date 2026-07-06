"""
Descomposición de Fourier de una señal de ECG — con preprocesamiento
--------------------------------------------------------------------
Se aplica un filtro pasa-altas (0.5 Hz) para eliminar la deriva de
línea base antes de la FFT, revelando la frecuencia cardíaca real.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import butter, filtfilt

# ----------------------------------------------------
# 1. Cargar los datos
# ----------------------------------------------------
FS  = 360
ecg = electrocardiogram()
n   = len(ecg)
t   = np.arange(n) / FS

# ----------------------------------------------------
# 2. Filtrar la deriva de línea base (pasa-altas 0.5 Hz)
# ----------------------------------------------------
# Un filtro pasa-altas elimina todo lo que esté POR DEBAJO de la
# frecuencia de corte — en este caso, la deriva lenta (< 0.5 Hz).
# Lo que queda es la actividad eléctrica cardíaca real.
b, a   = butter(N=4, Wn=0.5 / (FS / 2), btype='high')
ecg_filtrado = filtfilt(b, a, ecg)   # filtfilt = sin desfase de fase

# ----------------------------------------------------
# 3. FFT sobre la señal filtrada
# ----------------------------------------------------
fft_vals   = np.fft.fft(ecg_filtrado)
freqs      = np.fft.fftfreq(n, d=1/FS)
mitad      = n // 2
frec_pos   = freqs[1:mitad]
amplitudes = np.abs(fft_vals[1:mitad]) * 2 / n

# ----------------------------------------------------
# 4. Visualización
# ----------------------------------------------------
seg    = 10
mask   = t <= seg
rango  = frec_pos <= 5

fig, ejes = plt.subplots(3, 1, figsize=(10, 11))

# (a) Señal original vs filtrada (primeros 10 s)
ejes[0].plot(t[mask], ecg[mask],
             color="#AAAACC", linewidth=0.8, label="Original (con deriva)")
ejes[0].plot(t[mask], ecg_filtrado[mask],
             color="#534AB7", linewidth=0.9, label="Filtrada (pasa-altas 0.5 Hz)")
ejes[0].set_title(f"Señal ECG — primeros {seg} s")
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].set_ylabel("Amplitud (mV)")
ejes[0].legend(fontsize=8)

# (b) Espectro 0–10 Hz de la señal FILTRADA
rango_b = frec_pos <= 10
ejes[1].plot(frec_pos[rango_b], amplitudes[rango_b],
             color="#1D9E75", linewidth=0.9)
ejes[1].set_title("Espectro FFT — señal filtrada (0 – 10 Hz)")
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].set_ylabel("Amplitud")

# Marcar el pico dominante
idx_dom  = np.argmax(amplitudes[rango_b])
f_dom    = frec_pos[rango_b][idx_dom]
a_dom    = amplitudes[rango_b][idx_dom]
ejes[1].annotate(
    f"{f_dom:.2f} Hz\n(≈ {f_dom*60:.0f} lpm)",
    xy=(f_dom, a_dom),
    xytext=(f_dom + 0.4, a_dom * 0.88),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9,
)

# (c) Zoom 0–5 Hz con bandas de referencia
ejes[2].plot(frec_pos[rango], amplitudes[rango],
             color="#D85A30", linewidth=1)
ejes[2].axvspan(0.5, 3.0, alpha=0.12, color="orange",
                label="Rango fisiológico cardíaco (30–180 lpm)")
ejes[2].axvspan(0.0, 0.5, alpha=0.10, color="red",
                label="Zona de deriva / ruido (< 0.5 Hz)")
ejes[2].set_title("Zoom: 0 – 5 Hz (rango cardíaco)")
ejes[2].set_xlabel("Frecuencia (Hz)")
ejes[2].set_ylabel("Amplitud")
ejes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("fourier_ecg_filtrado.png", dpi=150)
plt.show()

# ----------------------------------------------------
# 5. Frecuencias dominantes
# ----------------------------------------------------
top_n       = 5
idx_top     = np.argsort(amplitudes)[::-1][:top_n]
print(f"\nLas {top_n} frecuencias dominantes (señal filtrada):")
for i in idx_top:
    print(f"  {frec_pos[i]:.3f} Hz  →  {frec_pos[i]*60:.1f} lpm  "
          f"|  Amplitud = {amplitudes[i]:.4f}")