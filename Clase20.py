import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

T = 1
t = 0
St = 100 # Precio del subyacente
K = 50 # Precio del ejercicio
r = 0.1 #Tasa
sigma = 0.4 #Volatilidad

def d(St, K, r, sigma, T, t=0):
    tau = T-t
    d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d1 = d2 + sigma * np.sqrt(tau)
    return d1, d2

def f_delta(K, r, sigma, T, t=0):
    prices = np.linspace(30,120,100)
    deltas = []
    for i in prices:
        d1, _ = d(i, K, r, sigma, T) 
        delta = norm.cdf(d1)
        deltas.append(delta)

    plt.plot(prices, deltas, label= "Delta")
    plt.xlabel('Precio del subyacente')
    plt.ylabel('Sensibilidad Delta')
    plt.legend()
    plt.grid()
    plt.show()

f_delta(K, r, sigma, T)

def f_rho(St, K, sigma, T, t=0):
    r_s = np.linspace(.08,.12,100)
    rhos = []
    for i in r_s:
        tau = T - t
        d1, d2 = d(St, K, i, sigma, T) 
        N_d1 = norm.cdf(d1)
        N_d1_prime = norm.pdf(d1) 
        N_d2 = norm.cdf(d2)
        rho = K * tau * np.exp(-r * tau) * N_d2 / 100
        rhos.append(rho)

    plt.plot(r_s, rhos, label= "Rho")
    plt.xlabel('Tasa')
    plt.ylabel('Sensibilidad Rho')
    plt.legend()
    plt.grid()
    plt.show()

f_rho(St, K, sigma, T)

def f_theta(St, K, r, sigma, T):
    time = np.linspace(t,T,100)
    thetas = []
    for i in time:
        tau = T - t
        d1, d2 = d(St, K, r, sigma, i) 
        N_d1 = norm.cdf(d1)
        N_d1_prime = norm.pdf(d1) 
        N_d2 = norm.cdf(d2)
        theta = (-St * N_d1_prime * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * N_d2) / 365
        thetas.append(theta)

    plt.plot(time, thetas, label= "Theta")
    plt.xlabel('Tiempo')
    plt.ylabel('Sensibilidad Theta')
    plt.legend()
    plt.grid()
    plt.show()
f_theta(St, K, r, sigma, T)

def f_vega(St, K, r, T):
    v = np.linspace(.35,.45,100)
    vegas = []
    for i in v:
        tau = T - t
        d1, d2 = d(St, K, r, i, T) 
        N_d1 = norm.cdf(d1)
        N_d1_prime = norm.pdf(d1) 
        N_d2 = norm.cdf(d2)
        vega = St * N_d1_prime * np.sqrt(tau) / 100
        vegas.append(vega)

    plt.plot(v, vegas, label= "Vega")
    plt.xlabel('Volatilidad')
    plt.ylabel('Sensibilidad Vega')
    plt.legend()
    plt.grid()
    plt.show()
f_vega(St, K, r, T)



