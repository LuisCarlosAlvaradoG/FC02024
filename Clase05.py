import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

class StockAnalysis:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.close_prices = None
        self.returns = None
        self.log_returns = None
    
    def get_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.close_prices = self.data['Close']
        self.calculate_returns()
    
    def calculate_returns(self):
        self.returns = self.close_prices.pct_change().dropna()
        self.log_returns = np.log(self.close_prices / self.close_prices.shift(1)).dropna()
    
    def descriptive_statistics(self):
        rd = self.returns.describe()
        lrd = self.log_returns.describe()
        df = pd.DataFrame({
        'Returns': rd,
        'Log Returns': lrd 
        })
        return df
    
    def plot_histogram(self):
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        sns.histplot(self.returns, kde=True, color='blue')
        plt.title(f'{self.ticker} Returns Histogram')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        sns.histplot(self.log_returns, kde=True, color='red')
        plt.title(f'{self.ticker} Log Returns Histogram')
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def two_sample_t_test(self):
        t_stat, p_value = stats.ttest_ind(self.returns, self.log_returns, equal_var=True)
        return t_stat, p_value

    def levene_test(self):
        w_stat, p_value = stats.levene(self.log_returns, self.log_returns)
        return w_stat, p_value
    
    def maximum_likelihood(self):
        mu, std = stats.norm.fit(self.returns)
        mu_l, std_l = stats.norm.fit(self.log_returns)
        return mu, std, mu_l, std_l

ticker = "AAPL"
stock = StockAnalysis(ticker=ticker, start_date="2020-01-01", end_date="2023-01-01")
stock.get_data()

print(stock.descriptive_statistics())

stock.plot_histogram()

t_stat, p_value = stock.two_sample_t_test()
print(f"Two-sample T-test: t-stat = {t_stat}, p-value = {p_value}")

w_stat, p_value = stock.levene_test()
print(f"Levene test: w-stat = {w_stat}, p-value = {p_value}")

mu, std, mu_l, std_l = stock.maximum_likelihood()
print(f"Maximum Likelihood Estimation: mu = {mu}, std = {std}, mu_log = {mu_l}, std_log = {std_l}")


##############################################################################################3
def get_data(ticker, start, end):
    data = yf.download(ticker, start= start, end=end)
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

ticker = "AAPL"
returns, log_returns = get_data(ticker=ticker, start="2020-01-01", end="2023-01-01")

descriptive_statistics(returns, log_returns)

plot_histogram(returns, log_returns, ticker)

t_stat, p_value = two_sample_t_test(returns, log_returns)
print(f"Two-sample T-test: t-stat = {t_stat}, p-value = {p_value}")

w_stat, p_value = levene_test(returns, log_returns)
print(f"Levene test: w-stat = {w_stat}, p-value = {p_value}")

mu, std, mu_l, std_l = maximum_likelihood(returns, log_returns)
print(f"Maximum Likelihood Estimation: mu = {mu}, std = {std}, mu_log = {mu_l}, std_log = {std_l}")