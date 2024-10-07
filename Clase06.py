import numpy as np
import matplotlib.pyplot as plt

import random
import matplotlib.pyplot as plt

random_walk = np.random.normal(0,1,1000)
plt.figure()
plt.plot(random_walk)
plt.show()

plt.figure()
plt.plot(np.cumsum(random_walk))
plt.show()


def lanzar_moneda(num):
    puntuacion = 0
    caminata = [0]
    for _ in range(num):
        if np.random.randint(0, 2) == 0:# np.random.choice([0, 1], p=[0.5, 0.5])
            puntuacion += 1
        else:
            puntuacion -= 1 
        caminata.append(puntuacion) 
    return caminata
     
num = 500
caminate = lanzar_moneda(num) 
plt.figure(figsize=(12, 6))
plt.plot(caminate, linestyle='-', color='b')
plt.title('Caminata Aleatoria con Lanzamientos de Moneda')
plt.xlabel('Número de Lanzamientos')
plt.ylabel('Puntuación')
plt.show()
np.mean(caminate)
np.std(caminate)


sm = np.array([lanzar_moneda(num) for _ in range(100)])

plt.figure(figsize=(12, 6))
plt.plot(sm.T)
plt.title('Caminatas Aleatorias Simuladas')
plt.xlabel('Número de Lanzamientos')
plt.ylabel('Puntuación')
plt.grid(True)
plt.show()