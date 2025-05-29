# NASDAQ API KEY
DnbXL5xxamVWyzj4Fqyf

import random

def asignar_proyectos(equipos, proyectos, repetir=False):
    if not repetir and len(proyectos) < len(equipos):
        raise ValueError("No hay suficientes proyectos para asignar sin repetición.")

    asignaciones = {}
    proyectos_disponibles = proyectos.copy() if not repetir else proyectos

    for equipo in equipos:
        if repetir:
            proyecto_asignado = random.choice(proyectos)
        else:
            proyecto_asignado = random.choice(proyectos_disponibles)
            proyectos_disponibles.remove(proyecto_asignado)
        
        asignaciones[equipo] = proyecto_asignado
    
    return asignaciones

# Ejemplo de uso
equipos = ["Equipo 1", "Equipo 2", "Equipo 3", "Equipo 4", "Equipo 5", "Equipo 6"]
proyectos = ["Expansión", "Contracción", "Cierre temporal", "Stages", "Abandono", "New tech"]

# Asignación sin repetición
asignaciones_sin_repeticion = asignar_proyectos(equipos, proyectos, repetir=False)
print("Asignaciones sin repetición:", asignaciones_sin_repeticion)

# Asignación con repetición
asignaciones_con_repeticion = asignar_proyectos(equipos, proyectos, repetir=True)
print("Asignaciones con repetición:", asignaciones_con_repeticion)
