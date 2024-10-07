import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

price = 5
cost = 0.8
inv = 1800000
R = 0.03
r = np.log(1+R)
fx_cost = 40000

data = pd.read_csv(r"C:\Users\luill\OneDrive\Documentos\Clases ITESO\Finanzas Cuantitativas\Presentaciones\Data_OilCompany.csv")
data

profit_loss = (data * (price - cost)) - fx_cost
profit_loss.insert(0, 'Year 0', -inv)
profit_loss

def NPV(profit_loss, price, cost, fx_cost, R):

    npv_list = []
    tir_list = []

    for index, row in profit_loss.iterrows():
        npv = npf.npv(R, row)
        tir = npf.irr(row)
        npv_list.append(npv)
        tir_list.append(tir*100)

    result_df = pd.DataFrame({'NPV ($)': npv_list, 'TIR (%)': tir_list})

    pd.options.display.float_format = '{:,.2f}'.format

    return result_df

result_df = NPV(profit_loss, price, cost, fx_cost, R)
result_df

p_rf = ((result_df['TIR (%)'] > (R * 100)).mean()) * 100
print(f'The probability that the IRR is over the Risk-free rate is: {p_rf:.2f}%')

p_35 = ((result_df['TIR (%)'] > 35).mean())*100
print(f'The probability that the IRR is over 35$ is: {p_35:.2f}%')

p_2m = ((result_df['NPV ($)'] > 2_000_000).mean())*100
print(f'The probability that the project value is over 2M $ is: {p_2m:.2f}%')

p_20 = (((result_df['TIR (%)'] >= 10) & (result_df['TIR (%)'] <= 20)).mean())*100
print(f'The probability that the project value is between 10% and 20% is: {p_20:.2f}%')

# Libraries
import numpy as np
import pandas as pd

# Number of simulations
n = 10000

# Values
R0 = 50
bar_R = 20
sigma_R = np.sqrt(900)
dt = 1

# Generate values for R y Epsilon using a normal dist
R = pd.DataFrame(np.random.normal(loc=R0 + bar_R * (dt), scale=sigma_R * np.sqrt(dt), size=n))
R.columns = ['R']
R["R"].mean()
R["R"].var()

epsilon = pd.DataFrame(np.random.normal(loc=0, scale=1, size=n))
epsilon.columns = ['Epsilon']

# Calculate R value using original equation
R['Simulated_R'] = R0 + bar_R * (dt) + sigma_R * epsilon * np.sqrt(dt)

# Print results
# print("R Simulation:")
# print(R)

R['Simulated_R'].mean()
R['Simulated_R'].var()