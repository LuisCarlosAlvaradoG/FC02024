import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros iniciales
T = 1  # Tiempo de vencimiento (1 año)
t_vals = np.linspace(0, T, 100)  # Valores de t desde 0 hasta T
S0 = 100  # Precio inicial del activo
K = 50   # Precio de ejercicio
r = .1
sigma = .4

# Función para calcular f(t) para put y call
def black_scholes_ft(St_vals, K, r, sigma, T, t_vals):
    ft_vals = []
    for i, t in enumerate(t_vals):
        tau = T - t  # Tiempo restante hasta vencimiento
        St = St_vals[i]
        d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d1 = d2 + sigma * np.sqrt(tau)
        ft = St * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        ft_vals.append(ft)
    return np.array(ft_vals)

def d(St, K, T, t, r, sigma):
    tau = T - t
    d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d1 = d2 + sigma * np.sqrt(tau)
    return d1, d2

def f_delta(K, T, r, sigma, t = 0):
    prices = np.linspace(20,120,100)
    deltas = []
    for i in prices:
        d1, _ = d(i, K, T, t, r, sigma)
        N_d1 = norm.cdf(d1)
        delta = N_d1
        deltas.append(delta)
    
    plt.plot(prices, deltas, label='Delta ')
    plt.xlabel('Precio del activo subyacente')
    plt.ylabel('Sensibilidad de la opción respecto a Delta')
    plt.legend()
    plt.grid(True)
    plt.show()

f_delta(K, T, r, sigma)

def f_theta(St, K, T, r, sigma, t = 0):
    times = np.linspace(t,T,100)
    thetas = []
    for i in times:
        tau = i - t
        d1, d2 = d(St, K, i, t, r, sigma)
        N_d1_prime = norm.pdf(d1) 
        N_d2 = norm.cdf(d2)
        theta = (-St * N_d1_prime * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * N_d2) / 365
        thetas.append(theta)
    
    plt.plot(times, thetas, label='Theta ')
    plt.xlabel('Tiempo')
    plt.ylabel('Sensibilidad de la opción respecto a Theta')
    plt.legend()
    plt.grid(True)
    plt.show()

f_theta(S0, K, T, r, sigma)

def f_vega(St, K, T, r, t = 0):
    sigmas = np.linspace(.35,.45,100)
    vegas = []
    for i in sigmas:
        tau = T - t
        d1, _ = d(St, K, T, t, r, i)
        N_d1_prime = norm.pdf(d1) 
        vega = St * N_d1_prime * np.sqrt(tau) / 100
        vegas.append(vega)
    
    plt.plot(sigmas, vegas, label='Vega')
    plt.xlabel('Sigma')
    plt.ylabel('Sensibilidad de la opción respecto a Vega')
    plt.legend()
    plt.grid(True)
    plt.show()

f_vega(S0, K, T, r)

def f_rho(St, K, T, sigma, t = 0):
    rs = np.linspace(.35,.45,100)
    rhos = []
    for i in rs:
        tau = T - t
        d1, _ = d(St, K, T, t, i, sigma)
        N_d1_prime = norm.pdf(d1) 
        rho = St * N_d1_prime * np.sqrt(tau) / 100
        rhos.append(rho)
    
    plt.plot(rs, rhos, label='Rho ')
    plt.xlabel('Tasa')
    plt.ylabel('Sensibilidad de la opción respecto a Rho')
    plt.legend()
    plt.grid(True)
    plt.show()

f_rho(S0, K, T, sigma)

# Función para calcular las griegas con más valores cambiantes
# Parámetros iniciales
T = 1  # Tiempo de vencimiento (1 año)
t_vals = np.linspace(0, T, 100)  # Valores de t desde 0 hasta T
S0 = 100  # Precio inicial del activo
K = 50   # Precio de ejercicio
r = .1
sigma = .4

def simulate_stock_price(S0, r, sigma, T, t_vals):
    np.random.seed(42)  # Para reproducibilidad
    dt = t_vals[1] - t_vals[0]
    W = norm.rvs(0,1,100)
    W = np.cumsum(W)
    S = S0 * np.exp((r - sigma**2/2)*(dt) + sigma*np.sqrt(dt)*W)
    return np.array(S)

def greeks(St_vals, K, r, sigma, T, t_vals):
    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
    for i, t in enumerate(t_vals):
        tau = T - t
        St = St_vals[i]
        d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d1 = d2 + sigma * np.sqrt(tau)
        
        N_d1 = norm.cdf(d1)
        N_d1_prime = norm.pdf(d1) 
        N_d2 = norm.cdf(d2)
        # Delta
        delta = N_d1
        # Gamma
        gamma = N_d1_prime / (St * sigma * np.sqrt(tau))
        # Theta (por aproximación)
        theta = (-St * N_d1_prime * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * N_d2) / 365
        # Vega
        vega = St * N_d1_prime * np.sqrt(tau) / 100
        # Rho
        rho = K * tau * np.exp(-r * tau) * N_d2 / 100
        deltas.append(delta)
        gammas.append(gamma)
        thetas.append(theta)
        vegas.append(vega)
        rhos.append(rho)

    return np.array(deltas), np.array(gammas), np.array(thetas), np.array(vegas), np.array(rhos)

def plot_greeks(t_vals, deltas, gammas, thetas, vegas, rhos, call1, St_vals, K):
    fig, ax = plt.subplots(3, 2, figsize=(14, 10))
    
    # Delta vs Tiempo con eje dual
    ax1 = ax[0, 0]
    ax1.plot(t_vals, deltas, label='Delta', color='blue')
    ax2 = ax1.twinx()  # Eje dual para call y precio
    ax2.plot(t_vals, call1, label='Call', color='red')
    ax2.axhline(K, color='red', linestyle='--', label='Strike')
    ax2.plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax1.set_title('Delta vs Tiempo')
    ax1.set_xlabel('Tiempo t')
    ax1.set_ylabel('Delta', color='blue')
    ax2.set_ylabel('Call/Price', color='green')
    ax1.grid()

    # Gamma vs Tiempo con eje dual
    ax1 = ax[0, 1]
    ax1.plot(t_vals, gammas, label='Gamma', color='blue')
    ax2 = ax1.twinx()  # Eje dual para call y precio
    ax2.plot(t_vals, call1, label='Call', color='red')
    ax2.axhline(K, color='red', linestyle='--', label='Strike')
    ax2.plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax1.set_title('Gamma vs Tiempo')
    ax1.set_xlabel('Tiempo t')
    ax1.set_ylabel('Gamma', color='blue')
    ax2.set_ylabel('Call/Price', color='green')
    ax1.grid()

    # Theta vs Tiempo con eje dual
    ax1 = ax[1, 0]
    ax1.plot(t_vals, thetas, label='Theta', color='blue')
    ax2 = ax1.twinx()  # Eje dual para call y precio
    ax2.plot(t_vals, call1, label='Call', color='red')
    ax2.axhline(K, color='red', linestyle='--', label='Strike')
    ax2.plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax1.set_title('Theta vs Tiempo')
    ax1.set_xlabel('Tiempo t')
    ax1.set_ylabel('Theta', color='blue')
    ax2.set_ylabel('Call/Price', color='green')
    ax1.grid()

    # Vega vs Tiempo con eje dual
    ax1 = ax[1, 1]
    ax1.plot(t_vals, vegas, label='Vega', color='blue')
    ax2 = ax1.twinx()  # Eje dual para call y precio
    ax2.plot(t_vals, call1, label='Call', color='red')
    ax2.axhline(K, color='red', linestyle='--', label='Strike')
    ax2.plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax1.set_title('Vega vs Tiempo')
    ax1.set_xlabel('Tiempo t')
    ax1.set_ylabel('Vega', color='blue')
    ax2.set_ylabel('Call/Price', color='green')
    ax1.grid()

    # Rho vs Tiempo con eje dual
    ax1 = ax[2, 0]
    ax1.plot(t_vals, rhos, label='Rho', color='blue')
    ax2 = ax1.twinx()  # Eje dual para call y precio
    ax2.plot(t_vals, call1, label='Call', color='red')
    ax2.axhline(K, color='red', linestyle='--', label='Strike')
    ax2.plot(t_vals, St_vals, label='Price', color='green', linestyle='-.')
    ax1.set_title('Rho vs Tiempo')
    ax1.set_xlabel('Tiempo t')
    ax1.set_ylabel('Rho', color='blue')
    ax2.set_ylabel('Call/Price', color='green')
    ax1.grid()
    
    # Eliminar el último subplot vacío
    ax[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

St_vals = simulate_stock_price(S0, r, sigma, T, t_vals)
call_K1_vals = black_scholes_ft(St_vals, K, r, sigma, T, t_vals)
deltas, gammas, thetas, vegas, rhos = greeks(St_vals, K, r, sigma, T, t_vals)

plot_greeks(t_vals, deltas, gammas, thetas, vegas, rhos, call_K1_vals, St_vals, K)
'''
1. Delta (Δ):
Rango: Entre 0 y 1 para una call.
Interpretación: La delta mide cuánto cambiará el precio de la opción si el precio del subyacente aumenta en 1 unidad.
Cuándo sube: La delta aumenta a medida que el precio del subyacente sube o cuando la opción se acerca al vencimiento y está "in the money" (ITM).
Cuándo baja: La delta disminuye cuando el precio del subyacente baja o cuando se acerca el vencimiento y la opción está "out of the money" (OTM).
Utilidad: Proporciona una idea de cuánto se moverá el precio de la opción respecto al precio del subyacente. Una delta alta indica que el precio de la opción se moverá casi en proporción al subyacente.

2. Gamma (Γ):
Rango: Siempre positivo para las calls y mayor cuando la opción está cerca del precio de ejercicio (at-the-money, ATM).
Interpretación: Mide la tasa de cambio de la delta con respecto al precio del subyacente. Es decir, indica cuánto cambiará la delta si el precio del subyacente cambia.
Cuándo sube: La gamma es mayor cuando el precio del subyacente está cerca del precio de ejercicio y disminuye a medida que el subyacente se mueve lejos del precio de ejercicio, ya sea ITM o OTM.
Cuándo baja: Gamma disminuye a medida que la opción se aleja de estar ATM y conforme se acerca el vencimiento.
Utilidad: Gamma es importante para entender la sensibilidad de la delta. Ayuda a los traders a gestionar su riesgo de cambio en la delta, ya que una alta gamma significa que la delta puede cambiar rápidamente.

3. Theta (Θ):
Rango: Siempre negativo para una posición long call (disminuye el valor con el tiempo).
Interpretación: Mide la sensibilidad del precio de la opción a la disminución del tiempo (decadencia temporal). Theta indica cuánto disminuirá el precio de la opción por cada día que pase.
Cuándo sube: Theta se vuelve más negativo conforme se acerca el vencimiento, especialmente cuando la opción está ATM.
Cuándo baja: Theta es menos negativo cuando la opción tiene mucho tiempo hasta el vencimiento o cuando está ITM u OTM.
Utilidad: Indica cuánto se deteriora el valor de la opción con el paso del tiempo. Para una posición long call, una theta alta significa que el valor de la opción se está erosionando rápidamente debido a la proximidad del vencimiento.

4. Vega (ν):
Rango: Positivo para una posición long call.
Interpretación: Mide la sensibilidad del precio de la opción a cambios en la volatilidad implícita. Vega indica cuánto cambiará el precio de la opción si la volatilidad del subyacente cambia un 1%.
Cuándo sube: Vega aumenta cuando la volatilidad implícita sube, y es mayor cuando la opción está ATM.
Cuándo baja: Vega disminuye cuando la volatilidad baja, y se reduce conforme la opción se mueve lejos del precio de ejercicio y se acerca al vencimiento.
Utilidad: Vega es útil para predecir cómo el precio de la opción responderá a cambios en la volatilidad del subyacente. Una alta volatilidad aumenta el precio de la opción, lo cual es beneficioso para los tenedores de opciones long call.

5. Rho (ρ):
Rango: Positivo para una posición long call.
Interpretación: Mide la sensibilidad del precio de la opción a cambios en la tasa de interés libre de riesgo. Rho indica cuánto cambiará el precio de la opción si la tasa de interés cambia en 1%.
Cuándo sube: Rho aumenta cuando las tasas de interés suben.
Cuándo baja: Rho disminuye cuando las tasas de interés bajan.
Utilidad: Aunque Rho tiende a ser menos relevante en comparación con las otras griegas, es importante cuando las tasas de interés fluctúan. Un aumento en las tasas de interés incrementa el valor de una long call porque reduce el valor presente del precio de ejercicio que se pagaría en el futuro.

Resumen de la Utilidad:
Delta: Te ayuda a saber cómo cambiará el precio de la opción si el subyacente se mueve.
Gamma: Te advierte cuán rápido cambiará la delta.
Theta: Te informa sobre la pérdida de valor con el paso del tiempo.
Vega: Te permite entender el impacto de cambios en la volatilidad.
Rho: Te indica cómo afectarán los cambios en las tasas de interés al valor de la opción.

Delta (ya lo mencionaste):

Si la delta es 0.87, significa que por cada dólar que suba el subyacente, el precio de la opción aumentará en 87 centavos. Si baja $1, la opción perderá 87 centavos.
Gamma:

La gamma mide cómo cambia la delta cuando el precio del subyacente cambia. Si la gamma es, por ejemplo, 0.05, significa que por cada dólar que suba el subyacente, la delta aumentará en 0.05.
Si el subyacente baja $1, la delta disminuirá en 0.05. Esto te indica qué tan sensible es la delta a los cambios en el precio del subyacente.
Theta:

La theta mide la pérdida de valor de la opción a medida que pasa el tiempo, conocido como "decadencia temporal".
Si la theta es, por ejemplo, -0.03, significa que por cada día que pasa, la opción perderá 3 centavos de su valor, todo lo demás constante.
Vega:

La vega mide cuánto cambia el precio de la opción si la volatilidad implícita del subyacente cambia.
Si la vega es 0.12, significa que por cada punto porcentual que suba la volatilidad, el precio de la opción aumentará en 12 centavos. Si baja un 1%, el precio de la opción disminuirá en 12 centavos.
Rho:

La rho mide cuánto cambia el precio de la opción si la tasa de interés cambia.
Si la rho es 0.07, significa que por cada punto porcentual que suban las tasas de interés, el precio de la opción aumentará en 7 centavos. Si las tasas bajan un 1%, el precio de la opción disminuirá en 7 centavos.
'''