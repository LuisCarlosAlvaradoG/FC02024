# -*- coding: utf-8 -*-
"""
@author: Luis Alvarado
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#%%
def get_data(ticker: str, start: str):
    data = yf.download(ticker, start=start)
    close_prices = data['Close'].dropna()
    returns = close_prices.pct_change().dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return close_prices, returns, log_returns

#%%
def plot_price_and_log_returns(price: pd.Series, log_returns: pd.Series, ticker: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1) Precio vs tiempo
    axes[0].plot(price.index, price, color='navy', lw=1.5)
    axes[0].set_title(f'{ticker} – Precio de Cierre vs Tiempo', fontsize=14)
    axes[0].set_ylabel('Precio (USD)')
    
    # 2) Log-returns vs tiempo
    axes[1].plot(log_returns.index, log_returns, color='crimson', lw=1)
    axes[1].set_title(f'{ticker} – Log Returns vs Tiempo', fontsize=14)
    axes[1].set_ylabel('Log Return')
    axes[1].set_xlabel('Fecha')
    
    plt.tight_layout()
    plt.show()

#%%
def plot_price_and_simple_returns(price: pd.Series, returns: pd.Series, ticker: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1) Precio vs tiempo
    axes[0].plot(price.index, price, color='darkgreen', lw=1.5)
    axes[0].set_title(f'{ticker} – Precio de Cierre vs Tiempo', fontsize=14)
    axes[0].set_ylabel('Precio (USD)')
    
    # 2) Simple returns vs tiempo
    axes[1].plot(returns.index, returns, color='purple', lw=1)
    axes[1].set_title(f'{ticker} – Simple Returns vs Tiempo', fontsize=14)
    axes[1].set_ylabel('Simple Return')
    axes[1].set_xlabel('Fecha')
    
    plt.tight_layout()
    plt.show()

#%%
def split_series_in_three(series: pd.Series):

    n = len(series)
    size = n // 3
    
    seg1 = series.iloc[0 : size]
    seg2 = series.iloc[size : 2*size]
    seg3 = series.iloc[2*size : ]
    
    return [seg1, seg2, seg3]

#%%
def anova_three_samples(seg1: np.ndarray, seg2: np.ndarray, seg3: np.ndarray):
    f_stat, p_value = stats.f_oneway(seg1, seg2, seg3)
    return f_stat, p_value

#%%
def levene_three_samples(seg1: np.ndarray, seg2: np.ndarray, seg3: np.ndarray):
    w_stat, p_value = stats.levene(seg1, seg2, seg3)
    return w_stat, p_value

#%% 1) Especifica el ticker y fecha de inicio
ticker = "AMZN"
start_date = "2020-01-01"

#%% 2) Obtenemos close_prices, simple returns y log-returns
close_prices, simple_returns, log_returns = get_data(ticker, start_date)

#%% 3) PUNTO 2 (imagen): Graficar S_t y r_t vs t
#    (en este caso r_t es el 'log_return')
print("\n--- Graficando Precio y Log-Returns ---")
plot_price_and_log_returns(close_prices, log_returns, ticker)

#%% 4) PUNTO 3 (imagen) sobre S_t (precios)
print("\n--- Dividiendo serie de Precios en 3 bloques iguales ---")
segments_price = split_series_in_three(close_prices)

#%% 4.1) ANOVA para medias de los 3 bloques de S_t
f_stat_price, p_val_price = anova_three_samples(
    segments_price[0].values,
    segments_price[1].values,
    segments_price[2].values
)
print(f"ANOVA Precios:   F-stat = {f_stat_price[0]:.4f},  p-value = {p_val_price[0]:.4f}")
if p_val_price < 0.05:
    print("  → Rechazamos H0: Al menos una media difiere.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 medias difieran.")

#%% 4.2) Levene para varianzas de los 3 bloques de S_t
w_stat_price, p_val_lev_price = levene_three_samples(
    segments_price[0].values,
    segments_price[1].values,
    segments_price[2].values
)
print(f"Levene Precios:  W-stat = {w_stat_price[0]:.4f},  p-value = {p_val_lev_price[0]:.4f}")
if p_val_lev_price < 0.05:
    print("  → Rechazamos H0 de varianzas iguales: Al menos una varianza difiere.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 varianzas difieran.")

#%% 5) PUNTO 3 (imagen) sobre r_t (log-returns)
print("\n--- Dividiendo  Log-Returns en 3 bloques iguales ---")
segments_logr = split_series_in_three(log_returns)

#%% 5.1) ANOVA para medias de los 3 bloques de r_t
f_stat_logr, p_val_logr = anova_three_samples(
    segments_logr[0].values,
    segments_logr[1].values,
    segments_logr[2].values
)
print(f"ANOVA Log-Returns:   F-stat = {f_stat_logr[0]:.4f},  p-value = {p_val_logr[0]:.4f}")
if p_val_logr < 0.05:
    print("  → Rechazamos H0: Al menos una media difiere entre bloques de log-returns.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 medias de log-returns difieran.")

#%% 5.2) Levene para varianzas de los 3 bloques de r_t
w_stat_logr, p_val_lev_logr = levene_three_samples(
    segments_logr[0].values,
    segments_logr[1].values,
    segments_logr[2].values
)
print(f"Levene Log-Returns:  W-stat = {w_stat_logr[0]:.4f},  p-value = {p_val_lev_logr[0]:.4f}")
if p_val_lev_logr < 0.05:
    print("  → Rechazamos H0: Al menos una varianza difiere entre bloques de log-returns.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 varianzas de log-returns difieran.")

#%% 6) PUNTO 4 (imagen): repetir gráfico y pruebas pero con SIMPLE RETURNS R_t
print("\n--- Graficando Precio y Simple Returns (R_t) ---")
plot_price_and_simple_returns(close_prices, simple_returns, ticker)

print("\n--- Dividiendo SIMPLE RETURNS en 3 bloques iguales ---")
segments_ret = split_series_in_three(simple_returns)

#%% 6.1) ANOVA para medias de los 3 bloques de R_t
f_stat_ret, p_val_ret = anova_three_samples(
    segments_ret[0].values,
    segments_ret[1].values,
    segments_ret[2].values
)
print(f"ANOVA Simple Returns:   F-stat = {f_stat_ret[0]:.4f},  p-value = {p_val_ret[0]:.4f}")
if p_val_ret < 0.05:
    print("  → Rechazamos H0: Al menos una media difiere entre bloques de simple returns.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 medias de simple returns difieran.")

#%% 6.2) Levene para varianzas de los 3 bloques de R_t
w_stat_ret, p_val_lev_ret = levene_three_samples(
    segments_ret[0].values,
    segments_ret[1].values,
    segments_ret[2].values
)
print(f"Levene Simple Returns:  W-stat = {w_stat_ret[0]:.4f},  p-value = {p_val_lev_ret[0]:.4f}")
if p_val_lev_ret < 0.05:
    print("  → Rechazamos H0: Al menos una varianza difiere entre bloques de simple returns.")
else:
    print("  → No rechazamos H0: No hay evidencia para decir que las 3 varianzas de simple returns difieran.")


#%%
np.random.seed(0)

# Simulamos incrementos ~ N(0,1) para los tiempos t = 1, 2, 3, 4, 5
increments = np.random.normal(0, 1, size=5)

# Construimos el proceso Wiener: W0 = 0; Wt = suma acumulada de los incrementos
W = np.concatenate(([0], np.cumsum(increments)))

# Creamos el DataFrame para visualizar valores
df = pd.DataFrame({'W_t': W})

df