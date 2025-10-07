# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 18:14:44 2025

@author: Luis Alvarado
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
#%%
def get_data(ticker, start):
    data = yf.download(ticker, start)
    close_prices = data["Close"]
    returns = close_prices.pct_change().dropna()
    log_returns = np.log(close_prices/close_prices.shift(1)).dropna()
    
    return returns, log_returns
#%%
def descriptive_statistics(returns, log_returns):
    rd = returns.describe()
    lrd = log_returns.describe()
    
    df = pd.concat(
        {'Returns':rd, "Log Returns": lrd}, axis = 1)
    
    return df
#%%
def plot_histogram(returns, log_returns, ticker):
    plt.figure(figsize=(14, 7)) 
    
    plt.subplot(1, 2, 1)
    sns.histplot(returns, kde= True, color="black")
    plt.title(f'{ticker} Returns Histogram')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(log_returns, kde= True, color="black")
    plt.title(f'{ticker} Log Returns Histogram')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
#%%
def two_sample_t_test(returns, log_returns):
    t_stat, p_value = st.ttest_ind(returns, log_returns, equal_var=True)
    return t_stat, p_value

def levene_test(returns, log_returns):
    w_stat, p_value = st.levene(returns, log_returns)
    return w_stat, p_value

def maximum_likelihood(returns, log_returns):
    mu, std = st.norm.fit(returns)
    mu_l, std_l = st.norm.fit(log_returns)
    
    return mu, std, mu_l, std_l

#%%
ticker = 'TSM'
returns, log_returns = get_data(ticker, start='2020-01-01')
#%%
descriptive_statistics(returns, log_returns)
#%%
plot_histogram(returns, log_returns, ticker)
#%%
t_stat, p_value = two_sample_t_test(returns, log_returns)
print(f"Two-sample T-test: t_stat = {t_stat} , p_value {p_value}") 
# So p_value > alpha
# No se rechaza la H0
# No hay evidencia significativa suficiente 
# para decir que las medias difieran, es decir, las medias
# son iguales.
#%%
w_stat, p_value = levene_test(returns, log_returns)
print(f"Levene Test: w_stat = {w_stat} , p_value {p_value}") 
# So p_value > alpha
# No se rechaza la H0
# No hay evidencia significativa suficiente 
# para decir que las varianzas difieran, es decir, las varianzas
# son iguales.
#%%
mu, std, mu_l, std_l = maximum_likelihood(returns, log_returns)
print(f"Returns: mu = {mu} , std {std} | Log Returns: mu = {mu_l} , std {std_l}") 
#%%
print(mu_l)
print(mu - std**2/2)
#%%
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    
    #DISTRIBUTIONS = [st.gennorm,st.genexpon,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #    st.nct,st.norm,st.powerlognorm, st.uniform, st.poisson     ]
    DISTRIBUTIONS = [st.norm, st.uniform]#

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)
#%%
best_fit_distribution(returns)
#%%
best_fit_distribution(log_returns)