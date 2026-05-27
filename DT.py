"""
GENERADOR DE DATOS TEMPORALES CON ESTACIONALIDAD PRESERVADA Y RANGO CONTROLADO

Propósito:
    Generar datos mensuales sintéticos que mantengan los patrones estacionales de 
    datos históricos de comercio internacional, acotados a un rango específico (25-40).

Enfoque Técnico:
    Se utiliza descomposición temporal (Tendencia + Estacionalidad + Residuo) en lugar
    de modelos ARIMA/SARIMA tradicionales, porque:

    Ventajas para este caso específico:
    1. PRESERVACIÓN GARANTIZADA - La estacionalidad se replica exactamente
    2. CONTROL DE RANGO - Escalado final sin distorsión de patrones temporales
    3. SIMPLICIDAD - No requiere ajuste de hiperparámetros complejos (p,d,q,P,D,Q)

    Flujo del proceso:
    Datos Históricos → Descomposición → [Tendencia] + [Estacionalidad] + [Residuo]
                            ↓
    Generación → [Tendencia Nueva] + [Misma Estacionalidad] + [Residuo Similar]
                            ↓
    Escalado Min-Max → Ajuste a rango 25-40
                            ↓
    Datos Finales → Mismo patrón temporal + Rango controlado

Contexto de Datos:
    - Series mensuales de comercio internacional de frambuesas, moras y loganberries
    - Patrón estacional marcado: compras altas (junio-agosto), ventas altas (enero-mayo)
    - Datos históricos con valores en escala original (miles-millones)
    - Requerimiento: generar datos sintéticos en rango 25-40 manteniendo temporalidad
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Comercio-internacional-neto-de-Frambuesas-Moras-y-Loganberries-Frescas--Flujo-mensual-segun-datos-de-Cinta-de-Aduanas.csv')

ventas = df[df['Flow'] == 'Ventas internacionales'].copy()

ventas['Date'] = pd.to_datetime(ventas['Month'], format='%Y-%m')
ventas = ventas.sort_values('Date')

def generar_serie_estacional(historico, meses_a_generar=12, rango_objetivo=(25, 40)):
    fechas_completas = pd.date_range(
        start=historico['Date'].min(),
        end=historico['Date'].max(),
        freq='MS'
    )
    
    serie_completa = pd.DataFrame({'Date': fechas_completas})
    serie_completa = serie_completa.merge(historico[['Date', 'Trade Value']], on='Date', how='left')
    
    serie_completa = serie_completa.set_index('Date')
    serie_completa = serie_completa.asfreq('MS')
    
    serie_interpolada = serie_completa['Trade Value'].interpolate()
    
    try:
        descomposicion = seasonal_decompose(serie_interpolada, model='additive', period=12)
        
        tendencia = descomposicion.trend
        estacionalidad = descomposicion.seasonal
        residuo = descomposicion.resid
    except:
        print("Usando método alternativo de estacionalidad")
        serie_interpolada = serie_completa['Trade Value'].fillna(method='ffill').fillna(method='bfill')
        estacionalidad = crear_estacionalidad_manual(serie_interpolada)
        tendencia = serie_interpolada.rolling(window=12, min_periods=1).mean()
        residuo = serie_interpolada - tendencia - estacionalidad
    
    ultima_tendencia = tendencia.dropna().iloc[-1] if len(tendencia.dropna()) > 0 else serie_interpolada.mean()
    nueva_tendencia = [ultima_tendencia] * meses_a_generar
    
    if len(estacionalidad) >= 12:
        patron_estacional = estacionalidad.iloc[-12:].values
    else:
        patron_estacional = crear_estacionalidad_manual(serie_interpolada)
    
    nueva_estacionalidad = np.tile(patron_estacional, (meses_a_generar // 12) + 1)[:meses_a_generar]
    
    residuo_limpio = residuo.dropna()
    if len(residuo_limpio) > 0:
        nuevo_residuo = np.random.normal(
            residuo_limpio.mean(), 
            residuo_limpio.std() * 0.3,  
            meses_a_generar
        )
    else:
        nuevo_residuo = np.random.normal(0, serie_interpolada.std() * 0.1, meses_a_generar)
    
    datos_crudos = np.array(nueva_tendencia) + nueva_estacionalidad + nuevo_residuo
    
    scaler = MinMaxScaler(feature_range=rango_objetivo)
    datos_finales = scaler.fit_transform(datos_crudos.reshape(-1, 1)).flatten()
    
    return datos_finales

def crear_estacionalidad_manual(serie):
    if len(serie) < 12:
        return pd.Series([0] * 12)
    
    meses = serie.index.month
    estacionalidad = []
    for mes in range(1, 13):
        valores_mes = serie[meses == mes]
        if len(valores_mes) > 0:
            estacionalidad.append(valores_mes.mean() - serie.mean())
        else:
            estacionalidad.append(0)
    
    return pd.Series(estacionalidad)

datos_ventas = generar_serie_estacional(ventas, meses_a_generar=18, rango_objetivo=(25, 40))

ultima_fecha = ventas['Date'].max()
fechas_futuras = pd.date_range(
    start=ultima_fecha + pd.DateOffset(months=1),
    periods=18,
    freq='MS'
)

plt.figure(figsize=(15, 10))
plt.plot(fechas_futuras, datos_ventas, 'orange', label='Generado Ventas', alpha=0.8)
plt.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=40, color='gray', linestyle='--', alpha=0.5)
plt.title('Ventas Internacionales - Histórico vs Generado (Escala 25-40)')
plt.ylabel('Valor')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_generado = pd.DataFrame({
    'Date': fechas_futuras,
    'Flow': ['Ventas internacionales'] * 18,
    'Trade Value': datos_ventas
})

print(df_generado)
print(f"Ventas  - Min: {datos_ventas.min():.2f}, Max: {datos_ventas.max():.2f}, Mean: {datos_ventas.mean():.2f}")

# df_generado.to_csv('datos_generados_estacionales.csv', index=False)

import numpy as np
import scipy.stats as st
import warnings
import pandas as pd

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    
    DISTRIBUTIONS = [st.gennorm,st.genexpon,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
       st.nct,st.norm,st.powerlognorm, st.uniform, st.poisson, st.beta]
    # DISTRIBUTIONS = [st.norm, st.uniform]#

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

data = pd.read_excel("PFO2025.xlsx")

best_premium, params_best_premium = best_fit_distribution(data["Rend 80/20"])
best_crumble, params_best_crumble = best_fit_distribution(data["Rend Crumble"])
best_bloque, params_best_bloque = best_fit_distribution(data["Rend Bloque"])
best_desjugue, params_best_desjugue = best_fit_distribution(data["Desjugue"])
best_merma, params_best_merma = best_fit_distribution(data["Merma"])

print("Mejor distribución Rend 80/20:", best_premium, params_best_premium)
print("Mejor distribución Rend Crumble:", best_crumble, params_best_crumble)
print("Mejor distribución Rend Bloque:", best_bloque, params_best_bloque)
print("Mejor distribución Desjugue:", best_desjugue, params_best_desjugue)
print("Mejor distribución Merma:", best_merma, params_best_merma)

# Generar 52 datos aleatorios basados en las distribuciones ajustadas
# El rendimiento total debe sumar 100% considerando las mermas y desjugue. No puede haber negativos.
np.random.seed(42)  # Para reproducibilidad
rendimientos_generados = []
num_datos = 52

for _ in range(num_datos):
    # Generar valores individuales
    rend_premium = getattr(st, best_premium).rvs(*params_best_premium[:-2], loc=params_best_premium[-2], scale=params_best_premium[-1])
    rend_crumble = getattr(st, best_crumble).rvs(*params_best_crumble[:-2], loc=params_best_crumble[-2], scale=params_best_crumble[-1])
    rend_bloque = getattr(st, best_bloque).rvs(*params_best_bloque[:-2], loc=params_best_bloque[-2], scale=params_best_bloque[-1])
    desjugue = getattr(st, best_desjugue).rvs(*params_best_desjugue[:-2], loc=params_best_desjugue[-2], scale=params_best_desjugue[-1])
    merma = getattr(st, best_merma).rvs(*params_best_merma[:-2], loc=params_best_merma[-2], scale=params_best_merma[-1])
    
    # Crear array y asegurar no negativos
    componentes = np.array([rend_premium, rend_crumble, rend_bloque, desjugue, merma])
    componentes = np.maximum(componentes, 0)  # Eliminar negativos
    
    # Reescalar para que sumen 1 (100%)
    total = np.sum(componentes)
    if total > 0:
        componentes_escalados = componentes / total
    else:
        # Fallback: distribución uniforme si todo es cero
        componentes_escalados = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    rendimientos_generados.append({
        "Rend 80/20": componentes_escalados[0],
        "Rend Crumble": componentes_escalados[1],
        "Rend Bloque": componentes_escalados[2],
        "Desjugue": componentes_escalados[3],
        "Merma": componentes_escalados[4],
        "Rendimiento Neto": np.sum(componentes_escalados)
    })

df_generados = pd.DataFrame(rendimientos_generados)
print(df_generados)
# 

# Calcular el valor teórico de cada distribucion
teorico_premium = getattr(st, best_premium).mean(*params_best_premium[:-2], loc=params_best_premium[-2], scale=params_best_premium[-1])
teorico_crumble = getattr(st, best_crumble).mean(*params_best_crumble[:-2], loc=params_best_crumble[-2], scale=params_best_crumble[-1])
teorico_bloque = getattr(st, best_bloque).mean(*params_best_bloque[:-2], loc=params_best_bloque[-2], scale=params_best_bloque[-1])
teorico_desjugue = getattr(st, best_desjugue).mean(*params_best_desjugue[:-2], loc=params_best_desjugue[-2], scale=params_best_desjugue[-1])
teorico_merma = getattr(st, best_merma).mean(*params_best_merma[:-2], loc=params_best_merma[-2], scale=params_best_merma[-1])
total_teorico = teorico_premium + teorico_crumble + teorico_bloque + teorico_desjugue + teorico_merma

print(f"Valor teórico Rend 80/20: {teorico_premium/total_teorico}")
print(f"Valor teórico Rend Crumble: {teorico_crumble/total_teorico}")
print(f"Valor teórico Rend Bloque: {teorico_bloque/total_teorico}")
print(f"Valor teórico Desjugue: {teorico_desjugue/total_teorico}")
print(f"Valor teórico Merma: {teorico_merma/total_teorico}")