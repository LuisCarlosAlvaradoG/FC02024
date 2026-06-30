# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:44:23 2026

@author: Luis Alvarado
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as st
import warnings
#%%
def get_data(ticker, start):
    data = yf.download(ticker, start= start) # end = end
    close_prices = data["Close"]
    returns = close_prices.pct_change().dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    return returns, log_returns

#%%
def descriptive_statistics(returns, log_returns):
    rd = returns.describe()
    lrd = log_returns.describe()
    
    df = pd.concat({"Returns": rd, "Log Returns": lrd}, axis = 1)
    
    return df
#%%
def plot_histogram(returns, log_returns, ticker):
    plt.figure(figsize=(14,7))
    
    plt.subplot(1, 2, 1)
    sns.histplot(returns, kde = True, color = "blue")
    plt.title(f'{ticker} Returns Histogram')
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.histplot(log_returns, kde = True, color = "red")
    plt.title(f'{ticker} Log Returns Histogram')
    plt.xlabel("Log Returns")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show() 
#%%
def two_sample_t_test(returns, log_returns):
    t_stat, p_value = stats.ttest_ind(returns, log_returns, equal_var=True)
    return t_stat, p_value
#%%
def levene_test(returns, log_returns):
    w_stat, p_value = stats.levene(returns, log_returns)
    return w_stat, p_value
#%%
def maximum_likelihood(returns, log_returns):
    mu, std = stats.norm.fit(returns)
    l_mu, l_std = stats.norm.fit(log_returns) 
    
    return mu, std, l_mu, l_std    
#%%
ticker = "PLTR"
returns, log_returns = get_data(ticker = ticker, start = "2022-01-01")
#%%
descriptive_statistics(returns, log_returns)
#%%
plot_histogram(returns, log_returns, ticker)
#%%
t_stat, p_value = two_sample_t_test(returns, log_returns)
print(f"Two-sample T-test: t-stat = {t_stat}, p-value = {p_value}")
#Si p_value ≥ α
#→ No rechazas 𝐻0 
#No hay evidencia suficiente para decir que las medias difieran, así que podemos asumir que las medias son iguales.
#%%
w_stat, p_value = levene_test(returns, log_returns)
print(f"Levene test: w-stat = {w_stat}, p-value = {p_value}")
#Si p_value ≥ 0.05:
#→ No rechazamos 𝐻0
#No hay evidencia para descartar que las varianzas sean iguales.
#%%
mu, std, mu_l, std_l = maximum_likelihood(returns, log_returns)
print(f"Maximum Likelihood Estimation: mu = {mu}, std = {std}, mu_log = {mu_l}, std_log = {std_l}")
#%%
print(f"{mu - std**2 / 2} = {mu_l}")





