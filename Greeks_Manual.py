import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def call(St, K, r, sigma, T, t):
    tau = T - t
    d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d1 = d2 + sigma * np.sqrt(tau)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = St * N_d1 - K * np.exp(-r * tau) * N_d2
    N_d1_prime = norm.pdf(d1) 
    gamma = N_d1_prime / (St * sigma * np.sqrt(tau))
    return d1, d2, N_d1, N_d2, call_price, gamma


d1, d2, N_d1, N_d2, call_price, gamma = call(100, 100, 0.03, 0.2, 1, 0)
d1_, d2_, N_d1_, N_d2_, call_price_, gamma_ = call(101, 100, 0.03, 0.2, 1, 0)

print(call_price, call_price_, call_price_ - call_price, N_d1, gamma, N_d1 + gamma)

def greeks(St, K, r, sigma, T, t):
        tau = T - t
        d2 = (np.log(St / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d1 = d2 + sigma * np.sqrt(tau)
        N_d1 = norm.cdf(d1)
        N_d1_prime = norm.pdf(d1) 
        delta = N_d1
        gamma = N_d1_prime / (St * sigma * np.sqrt(tau))
        return delta, gamma
