# -*- coding: utf-8 -*-
"""
@author: Luis Alvarado
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

np.random.seed(42)
#%% Creamos valores aleatorios a partir de una
# distribución
risk_free = .03
alfa = np.random.normal(.15, .08, 1000)
beta = 0.9
market_return = np.random.uniform(.03,.21,1000)

#%% Modelo CAPM

CAPM = alfa + beta*(market_return - risk_free) + risk_free

#Visualizar los resultados
sns.histplot(CAPM, kde=True)
plt.title("Distribución del Retorno Esperado de la Acción")
plt.axvline(CAPM.mean(), color='red', linestyle='--')
plt.xlabel("Retorno esperado")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()
#%%
CAPM.mean(), CAPM.var(), CAPM.std()
#%% Ajustar una distribución a mis datos ya conocidos
params = st.t.fit(CAPM)
arg = params[:-2]
loc = params[-2]
scale = params[-1]

# Esta es una forma de comprobar que tan bien
# se ajustan nuestros datos a la distribución. 
# Pero hay muchos test para comprobarlos

y, x = np.histogram(CAPM, bins=100, density = True)
x = (x + np.roll(x, -1))[:-1] / 2.0
pdf = st.t.pdf(x, loc = loc, scale = scale, *arg)
sse = np.sum(np.power(y - pdf, 2.0))