import numpy as np
import pandas as pd

alumnos = ["Luis","Mateo", "Fer", "Arturo", "Chuy", "Eleazar", "Dayanni", "Michelle", "Paula", "Max"]
selecciones = ["México", "Corea del Sur", "R. Checa", "Sudafrica", "Suiza", "Canadá", "Qatar", "B&H", "Escocia", "Marruecos", "Brasil", "Haiti",
               "USA", "Australia", "Turquia", "Paraguay", "Alemania", "Costa de Marfil", "Ecuador", "Curazao", "Suecia", "Japón", "Países Bajos", "Túnez",
               "Nueva Zelanda", "Iran", "Bélgica", "Egipto", "Uruguay", "Arabia Saudita", "España", "Cabo Verde", "Noruega", "Francia", "Senegal", "Iraq",
               "Argentina", "Austria", "Jordania", "Argelia", "Portugal", "RD Congo", "Uzbekistan", "Colombia", "Inglaterra", "Croacia", "Ghana", "Panamá"] 

# Asigna aleatoriamente n selecciones a cada alumno dado n = len(selecciones) // len(alumnos) sin repetir selecciones entre alumnos,
# si sobran selecciones, asignarlas a los alumnos de forma aleatoria sin repetir selecciones entre alumnos
n = len(selecciones) // len(alumnos)
res = len(selecciones) % len(alumnos)
selecciones_asignadas = {}
selecciones_disponibles = selecciones.copy()

for alumno in alumnos:
    selecciones_asignadas[alumno] = []
    for i in range(int(n)):
        seleccion = np.random.choice(selecciones_disponibles)
        selecciones_asignadas[alumno].append(seleccion)
        selecciones_disponibles.remove(seleccion)

alumnos_res = np.random.choice(alumnos, size = int(res), replace = False)
for i in alumnos_res:
    seleccion = np.random.choice(selecciones_disponibles)
    selecciones_asignadas[i].append(seleccion)
    selecciones_disponibles.remove(seleccion)

quiniela = pd.DataFrame.from_dict(selecciones_asignadas, orient='index').transpose().fillna("")
print(quiniela)