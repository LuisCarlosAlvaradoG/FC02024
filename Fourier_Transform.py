"""
Descomposición de Fourier del precio de un activo financiero
--------------------------------------------------------------
Descarga el precio de cierre de un activo desde Yahoo Finance y aplica
la Transformada de Fourier (FFT) para identificar los ciclos/frecuencias
que componen la serie de precios.

Requisitos (instalar una sola vez):
    pip install yfinance numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------------------------------
# 1. Descargar los datos
# ----------------------------------------------------
TICKER = "AAPL"     # cambia el ticker: "MSFT", "TSLA", "BTC-USD", etc.
PERIODO = "2y"      # rango: "6mo", "1y", "2y", "5y", "max"

data = yf.download(TICKER, period=PERIODO, progress=False)
precio = data["Close"].dropna()
log_returns = np.abs(np.log(precio / precio.shift(1)))
n = len(precio)

# ----------------------------------------------------
# 3. Aplicar la Transformada de Fourier (FFT)
# ----------------------------------------------------
fft_valores = np.fft.fft(log_returns)
frecuencias = np.fft.fftfreq(n, d=1)  # d=1 -> una muestra por día

# Nos quedamos solo con la mitad positiva del espectro
# (la mitad negativa es un espejo de la positiva, no aporta info nueva)
mitad = n // 2
frecuencias_pos = frecuencias[1:mitad]
amplitudes = np.abs(fft_valores[1:mitad]) * 2 / n
periodos = 1 / frecuencias_pos  # período del ciclo, en días

# ----------------------------------------------------
# 4. Visualización
# ----------------------------------------------------
fig, ejes = plt.subplots(3, 1, figsize=(10, 11))

# (a) Señal original en el tiempo
ejes[0].plot(data.index, log_returns, color="#534AB7", linewidth=1.2)
ejes[0].set_title(f"{TICKER}: precio de cierre (dominio del tiempo)")
ejes[0].set_xlabel("Fecha")
ejes[0].set_ylabel("Precio (USD)")

# (b) Espectro de frecuencias
ejes[1].plot(frecuencias_pos, amplitudes, color="#1D9E75", linewidth=1)
ejes[1].set_title("Espectro de frecuencias (FFT)")
ejes[1].set_xlabel("Frecuencia (ciclos por día)")
ejes[1].set_ylabel("Amplitud")

# (c) Mismo espectro, en función del período (más fácil de interpretar)
ejes[2].plot(periodos, amplitudes, color="#D85A30", linewidth=1)
ejes[2].set_xlim(0, 250)  # nos enfocamos en ciclos de hasta ~250 días
ejes[2].set_title("Espectro en función del período (días por ciclo)")
ejes[2].set_xlabel("Período (días)")
ejes[2].set_ylabel("Amplitud")

plt.tight_layout()
plt.savefig("fourier_activo.png", dpi=150)
plt.show()

# ----------------------------------------------------
# 5. Imprimir las frecuencias dominantes
# ----------------------------------------------------
top_n = 5
indices_top = np.argsort(amplitudes.flatten())[::-1][:top_n]
print(f"\nLas {top_n} frecuencias dominantes en {TICKER}:")
for i in indices_top:
    print(f"  Período ≈ {periodos[i]:.2f} días | Amplitud = {amplitudes.flatten()[i]:.4f}")