import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as st
import numpy as np
import warnings

def get_data(ticker, start):
    data = yf.download(ticker, start= start)#,end=end)
    close_prices = data['Close']
    returns = close_prices.pct_change().dropna()
    # log_returns = np.log(returns).dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    return returns, log_returns

def descriptive_statistics(returns, log_returns):
    rd = returns.describe()
    lrd = log_returns.describe()
    df = pd.DataFrame({
    'Returns': rd,
    'Log Returns': lrd 
    })
    return df

def plot_histogram(returns, log_returns, ticker):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sns.histplot(returns, kde=True, color='blue')
    plt.title(f'{ticker} Returns Histogram')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(log_returns, kde=True, color='red')
    plt.title(f'{ticker} Log Returns Histogram')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def two_sample_t_test(returns, log_returns):
    t_stat, p_value = stats.ttest_ind(returns, log_returns, equal_var=True)
    return t_stat, p_value

def levene_test(returns, log_returns):
    w_stat, p_value = stats.levene(returns, log_returns)
    return w_stat, p_value

def maximum_likelihood(returns, log_returns):
    mu, std = stats.norm.fit(returns)
    mu_l, std_l = stats.norm.fit(log_returns)
    return mu, std, mu_l, std_l

ticker = "FRAGUAB.MX"
returns, log_returns = get_data(ticker=ticker, start='2014-09-03')

descriptive_statistics(returns, log_returns)

plot_histogram(returns, log_returns, ticker)

t_stat, p_value = two_sample_t_test(returns, log_returns)
print(f"Two-sample T-test: t-stat = {t_stat}, p-value = {p_value}")

w_stat, p_value = levene_test(returns, log_returns)
print(f"Levene test: w-stat = {w_stat}, p-value = {p_value}")

mu, std, mu_l, std_l = maximum_likelihood(returns, log_returns)
print(f"Maximum Likelihood Estimation: mu = {mu}, std = {std}, mu_log = {mu_l}, std_log = {std_l}")

mu_l , mu - (std**2/2)

def best_fit_distribution(data, bins=200, ax=None):
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    
    DISTRIBUTIONS = [st.norm, st.uniform]

    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    for distribution in DISTRIBUTIONS:

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                params = distribution.fit(data)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                
                except Exception:
                    pass

                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

best_fit_distribution(returns)
best_fit_distribution(log_returns)