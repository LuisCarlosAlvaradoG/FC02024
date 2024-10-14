import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros iniciales
T = 1  # Tiempo de vencimiento (1 año)
t_vals = np.linspace(0, T, 100)  # Valores de t desde 0 hasta T
S0 = 100  # Precio inicial del activo
K1 = 50   # Precio de ejercicio 1
K2 = 150  # Precio de ejercicio 2
r = 0.1   # Ajusto r a un valor razonable (10%)
sigma = 0.4  # Volatilidad

# Generar una función que simula la evolución de St
def simulate_stock_price(S0, r, sigma, T, t_vals):
    np.random.seed(42)  # Para reproducibilidad
    dt = t_vals[1] - t_vals[0]
    W = norm.rvs(0,1,100)
    W = np.cumsum(W)
    S = S0 * np.exp((r - sigma**2/2)*(dt) + sigma*np.sqrt(dt)*W)
    return np.array(S)

# Función para calcular f(t) para put y call
def black_scholes_ft(St_vals, K, r, sigma, T, t_vals, option_type='call'):
    ft_vals = []
    for i, t in enumerate(t_vals):
        tau = T - t  # Tiempo restante hasta vencimiento
        St = St_vals[i]
        d2 = (np.log(St / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d1 = d2 + sigma * np.sqrt(tau)
        
        if option_type == 'call':
            ft = St * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        elif option_type == 'put':
            ft = K * np.exp(-r * tau) * norm.cdf(-d2) - St * norm.cdf(-d1)
        elif option_type == 'call_cash':
            ft = K * np.exp(-r * tau) * norm.cdf(d2)
        elif option_type == 'put_cash':
            ft = K * np.exp(-r * tau) * norm.cdf(-d2)
        elif option_type == 'call_asset':
            ft = St * norm.cdf(d1)
        elif option_type == 'put_asset':
            ft = St * norm.cdf(-d1)
        
        ft_vals.append(ft)
    return np.array(ft_vals)

# Simular los precios del activo subyacente
St_vals = simulate_stock_price(S0, r, sigma, T, t_vals)

# Cálculo del valor de las opciones put y call en función del tiempo
call_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'call')
put_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'put')

call_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'call')
put_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'put')

# Cash-or-Nothing Call y Put
call_cash_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'call_cash')
put_cash_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'put_cash')

call_cash_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'call_cash')
put_cash_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'put_cash')

# Asset-or-Nothing Call y Put
call_asset_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'call_asset')
put_asset_K1_vals = black_scholes_ft(St_vals, K1, r, sigma, T, t_vals, 'put_asset')

call_asset_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'call_asset')
put_asset_K2_vals = black_scholes_ft(St_vals, K2, r, sigma, T, t_vals, 'put_asset')

def graph(t_vals, St_vals, S0, K1, K2, call1, call2, put1, put2, title1, title2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Long Call
    ax[0].plot(t_vals, call1, label='K=50 (Call)', color='blue')
    ax[0].axhline(K1, color='blue', label='Strike')
    ax[0].plot(t_vals, call2, label='K=150 (Call)', color='red', linestyle='--')
    ax[0].axhline(K2, color='red', linestyle='--', label='Strike')
    ax[0].plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax[0].scatter(0, S0, color='green', marker='*', s=100, label='S0')
    ax[0].set_title(f'{title1} (Valor vs Tiempo)')
    ax[0].set_xlabel('Tiempo t')
    ax[0].set_ylabel('Valor de la Opción')
    ax[0].legend()
    # Long Put
    ax[1].plot(t_vals, put1, label='K=50 (Put)', color='blue')
    ax[1].axhline(K1, color='blue', label='Strike')
    ax[1].plot(t_vals, put2, label='K=150 (Put)', color='red', linestyle='--')
    ax[1].axhline(K2, color='red', linestyle='--', label='Strike')
    ax[1].plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax[1].scatter(0, S0, color='green', marker='*', s=100, label='S0')
    ax[1].set_title(f'{title2} (Valor vs Tiempo)')
    ax[1].set_xlabel('Tiempo t')
    ax[1].set_ylabel('Valor de la Opción')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

graph(t_vals, St_vals, S0, K1, K2, call_K1_vals, call_K2_vals, put_K1_vals, put_K1_vals, "Call European Financial Options", "Put European Financial Options")
graph(t_vals, St_vals, S0, K1, K2, call_cash_K1_vals, call_cash_K2_vals, put_cash_K1_vals, put_cash_K2_vals, "Call European Binary Options Cash or Nothing", "Put European Binary Options Cash or Nothing") 
graph(t_vals, St_vals, S0, K1, K2, call_asset_K1_vals, call_asset_K2_vals, put_asset_K1_vals, put_asset_K2_vals, "Call European Binary Options Asset or Nothing", "Put European Binary Options Asset or Nothing")
