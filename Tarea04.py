import numpy as np
import pandas as pd
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings

np.random.seed(42)
#%%
rf = 0.03
price = 5
cost = 0.8
initial_inv = 1800000
op_cost = 40000
#%%
df = pd.read_csv('Data_OilCompany.csv')
df.head()
#%%
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    
    DISTRIBUTIONS = [st.norm, st.uniform,]

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
Year_1 = best_fit_distribution(data=df['  Year 1  '])
Year_1
#%%
Year_2 = best_fit_distribution(data=df['  Year 2  '])
Year_2
#%%
Year_3 = best_fit_distribution(data=df['  Year 3  '])
Year_3
#%%
Year_4 = best_fit_distribution(data=df['  Year 4  '])
Year_4
#%%
Year_5 = best_fit_distribution(data=df['  Year 5  '])
Year_5
#%% PREGUNTA 1
mu, sigma = Year_1[1]
mean_Y1 = st.norm.mean(loc=mu, scale=sigma)
var_Y1  = st.norm.var (loc=mu, scale=sigma)

a, scale = Year_2[1]
mean_Y2 = st.uniform.mean(loc=a, scale=scale)
var_Y2  = st.uniform.var (loc=a, scale=scale)

a, scale = Year_3[1]
mean_Y3 = st.uniform.mean(loc=a, scale=scale)
var_Y3  = st.uniform.var (loc=a, scale=scale)

mu, sigma = Year_4[1]
mean_Y4 = st.norm.mean(loc=mu, scale=sigma)
var_Y4  = st.norm.var (loc=mu, scale=sigma)

mu, sigma = Year_5[1]
mean_Y5 = st.norm.mean(loc=mu, scale=sigma)
var_Y5  = st.norm.var (loc=mu, scale=sigma)
#%% PREGUNTA 2
cashflow = df * (price - cost) - op_cost
cashflow.insert(0, 'Year 0', - initial_inv)

irr = cashflow.copy()
irr['IRR'] = irr.apply(lambda row: npf.irr(row), axis=1)
irr = irr['IRR']

npv = cashflow.copy()
npv['NPV'] = npv.apply(lambda row: npf.npv(rf,row), axis=1)
npv = npv['NPV']

irr_fit = best_fit_distribution(data=irr)
npv_fit = best_fit_distribution(data=npv)

mu, sigma = irr_fit[1]
mean_IRR = st.norm.mean(loc=mu, scale=sigma)
var_IRR  = st.norm.var (loc=mu, scale=sigma)

mu, sigma = npv_fit[1]
mean_NPV = st.norm.mean(loc=mu, scale=sigma)
var_NPV  = st.norm.var (loc=mu, scale=sigma)
#%% PREGUNTA 3
E_NPV = - initial_inv + (mean_Y1*(price - cost) - op_cost)/(1+rf)**(1) + (mean_Y2*(price - cost) - op_cost)/(1+rf)**(2) + (mean_Y3*(price - cost) - op_cost)/(1+rf)**(3) + (mean_Y4*(price - cost) - op_cost)/(1+rf)**(4) + (mean_Y5*(price - cost) - op_cost)/(1+rf)**(5)
VAR_NPV = ((price-cost)/(1+rf)**(1))**(2)*var_Y1 + ((price-cost)/(1+rf)**(2))**(2)*var_Y2 + ((price-cost)/(1+rf)**(3))**(2)*var_Y3 + ((price-cost)/(1+rf)**(4))**(2)*var_Y4 + ((price-cost)/(1+rf)**(5))**(2)*var_Y5
#%% PREGUNTA 4
# 1)
1 - st.norm.cdf(rf, loc=irr_fit[1][0], scale=irr_fit[1][1])
#%% 2)
1 - st.norm.cdf(0.35, loc=irr_fit[1][0], scale=irr_fit[1][1])
#%% 3)
1 - st.norm.cdf(2000000, loc=npv_fit[1][0], scale=npv_fit[1][1])
#%% 4)
st.norm.cdf(0.2, loc=irr_fit[1][0], scale=irr_fit[1][1]) - st.norm.cdf(0.1, loc=irr_fit[1][0], scale=irr_fit[1][1])
