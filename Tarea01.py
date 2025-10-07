import pandas as pd
import numpy as np
import numpy_financial as npf

df = pd.read_csv('Data_OilCompany.csv')
df.head()

rf = 0.03 #tasa de descuento
price = 5
cost = 0.8
initial_inv = 1800000
op_cost = 40000

cashflow = df * (price - cost) - op_cost
cashflow.insert(0, 'Year 0', -initial_inv)
cashflow.head()

irr = cashflow.copy()
irr['IRR'] = irr.apply(lambda row: npf.irr(row), axis=1)
irr = irr['IRR']
irr.head()

npv = cashflow.copy()
npv['NPV'] = npv.apply(lambda row: npf.npv(rf,row), axis=1)
npv = npv['NPV']
npv.head()

# Prob IRR >= 3%
(irr >= 0.03).mean()
# Prob IRR >= 35%
(irr >= 0.35).mean()
# Prob 10% <= IRR <= 20%
((irr >= 0.10) & (irr <= 0.20)).mean()
# Prob NPV >= 2M
(npv >= 2000000).mean()

def cash_flow(rf, price, costo, initial_investment, operational_cost, data, t):
    # 1) DataFrame de cashflows con mismas filas
    data_cashflow = pd.DataFrame(index=data.index)
    data_cashflow['Year 0'] = -float(initial_investment)

    # 2) Flujos por año: unidades * (price - costo) - operational_cost
    margin = float(price) - float(costo)
    for i in range(1, t+1):
        unidades = pd.to_numeric(data.iloc[:, i-1], errors="coerce").fillna(0.0)
        data_cashflow[f'Year {i}'] = unidades * margin - float(operational_cost)

    # 3) TIR por fila (en decimales: 0.15 = 15%)
    data_cashflow['%IRR'] = np.round(data_cashflow.apply(
        lambda row: float(npf.irr(row.values.astype(float))) ,axis=1),4)*100

    # --- NPV con tasa rf ---
    cols = [f'Year {i}' for i in range(0, t+1)]
    disc = np.array([(1+rf)**(-i) for i in range(0, t+1)], dtype=float)
    data_cashflow['NPV'] = np.rint(data_cashflow[cols].values * disc).sum(axis=1).astype(int)

    return data_cashflow

data_cashflow = cash_flow(rf=0.03, price=5, costo=0.80,
                          initial_investment=1_800_000,
                          operational_cost=40_000,
                          data=df, t=5)

data_cashflow

irr_2 = data_cashflow['%IRR'].dropna()
npv_2 = data_cashflow['NPV'].dropna()
rf = 0.03
p1 = (irr > rf).mean()                           # IRR > riesgo libre
p2 = (irr > 0.35).mean()                         # IRR > 35%
p3 = (npv > 2_000_000).mean()                    # NPV > $2M
p4 = ((irr >= 0.10) & (irr <= 0.20)).mean()      # 10% <= IRR <= 20%

print(f"1) P(IRR > rf={rf:.2%}): {p1:.2%}")
print(f"2) P(IRR > 35%):        {p2:.2%}")
print(f"3) P(NPV > $2M):        {p3:.2%}")
print(f"4) P(10% ≤ IRR ≤ 20%):  {p4:.2%}")