import numpy as np
import matplotlib.pyplot as plt


S0 = 100 
steps = 1000 
random_changes = np.random.normal(0, 1, steps)
prices = np.zeros(steps)
prices[0] = S0  

for t in range(1, steps):
    prices[t] = prices[t-1] + random_changes[t]

plt.figure(figsize=(10, 6))
plt.plot(random_changes, label='Caminata Aleatoria')
plt.title('Simulación de una Caminata Aleatoria')
plt.xlabel('Tiempo (Pasos)')
plt.ylabel('Rendimientos')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(prices, label='Caminata Aleatoria del Precio')
plt.title('Simulación de una Caminata Aleatoria')
plt.xlabel('Tiempo (Pasos)')
plt.ylabel('Precio del Activo')
plt.legend()
plt.show()
