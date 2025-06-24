import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros
T = 1.0
N = 100
t_vals = np.linspace(0, T, N+1)     # Incluimos T, pero evitamos tau=0
t_vals = t_vals[:-1]                # quitamos el último para no tener tau=0
S0 = 100.0
K1, K2 = 50.0, 150.0
r = 0.10
sigma = 0.40
np.random.seed(42)
#%%
def simulate_stock_price(S0, r, sigma, t_vals):
    dt = t_vals[1] - t_vals[0]
    # Generamos incrementos Z ~ N(0,1)
    Z = np.random.randn(len(t_vals))
    W = np.concatenate(([0.0], np.cumsum(Z))) * np.sqrt(dt)
    # Ahora construimos la trayectoria
    S = S0 * np.exp((r - 0.5*sigma**2)*np.concatenate(([0.0], t_vals)) + sigma * W)
    return S[:-1]  # devolvemos long = len(t_vals)
#%%
def black_scholes_ft(St, K, r, sigma, t_vals, option_type='call'):
    tau = T - t_vals
    # evitar tau=0:
    tau = np.maximum(tau, 1e-10)
    d1 = (np.log(St/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    if option_type == 'call':
        return St*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    if option_type == 'put':
        return K*np.exp(-r*tau)*norm.cdf(-d2) - St*norm.cdf(-d1)
    if option_type == 'call_cash':
        return K*np.exp(-r*tau)*norm.cdf(d2)
    if option_type == 'put_cash':
        return K*np.exp(-r*tau)*norm.cdf(-d2)
    if option_type == 'call_asset':
        return St*norm.cdf(d1)
    if option_type == 'put_asset':
        return St*norm.cdf(-d1)
    raise ValueError("option_type inválido")

#%% Simulación
St_vals = simulate_stock_price(S0, r, sigma, t_vals)

#%% Precios
call_K1 = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'call')
call_K2 = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'call')
put_K1  = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'put')
put_K2  = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'put')

call_cash_K1 = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'call_cash')
call_cash_K2 = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'call_cash')
put_cash_K1  = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'put_cash')
put_cash_K2  = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'put_cash')

call_asset_K1 = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'call_asset')
call_asset_K2 = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'call_asset')
put_asset_K1  = black_scholes_ft(St_vals, K1, r, sigma, t_vals, 'put_asset')
put_asset_K2  = black_scholes_ft(St_vals, K2, r, sigma, t_vals, 'put_asset')

#%%
def graph(t, S, K1, K2, c1, c2, p1, p2, title1, title2):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    # Call
    ax = axes[0]
    ax.plot(t, c1, label=f'Call K={K1}')
    ax.plot(t, c2, '--', label=f'Call K={K2}')
    ax.plot(t, S, '-.', label='S(t)')
    ax.axhline(K1, color='gray', lw=1)
    ax.axhline(K2, color='gray', lw=1, linestyle='--')
    ax.set_title(title1); ax.legend()
    # Put
    ax = axes[1]
    ax.plot(t, p1, label=f'Put K={K1}')
    ax.plot(t, p2, '--', label=f'Put K={K2}')
    ax.plot(t, S, '-.', label='S(t)')
    ax.axhline(K1, color='gray', lw=1)
    ax.axhline(K2, color='gray', lw=1, linestyle='--')
    ax.set_title(title2); ax.legend()
    plt.tight_layout()
    plt.show()
    
#%%
graph(t_vals, St_vals, K1, K2,
      call_K1, call_K2, put_K1, put_K2,
      "European Financial: Call vs Tiempo",
      "European Financial: Put vs Tiempo")

graph(t_vals, St_vals, K1, K2,
      call_cash_K1, call_cash_K2, put_cash_K1, put_cash_K2,
      "Binary Cash-or-Nothing Call",
      "Binary Cash-or-Nothing Put")

graph(t_vals, St_vals, K1, K2,
      call_asset_K1, call_asset_K2, put_asset_K1, put_asset_K2,
      "Binary Asset-or-Nothing Call",
      "Binary Asset-or-Nothing Put")
