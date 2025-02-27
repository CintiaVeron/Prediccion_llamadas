import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Generar el Rango de Fechas

# Definimos  fechas

start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)

# Generamos un rango de fechas diarias
dates = pd.date_range(start=start_date, end=end_date, freq='D')


# Simulacion del Volumen de Llamadas 

# Base de llamadas con una media de 1000 llamadas/día
np.random.seed(42)  # Para reproducibilidad
calls = 1000 + 200 * np.sin(np.linspace(0, 50, len(dates)))  # Componente estacional
calls += np.random.randint(-100, 100, len(dates))  # Ruido aleatorio

# Ajustamos por día de la semana (menos llamadas los fines de semana)
calls = np.where(pd.Series(dates).dt.dayofweek >= 5, calls * 0.8, calls)  # 20% menos sábados y domingos
calls = calls.astype(int)  # Convertir a enteros

# Simulacion el TMO (Tiempo Medio de Operación)  
# Generamos  TMO entre 4 y 6 minutos con variación diaria

tmo = np.random.uniform(4, 6, len(dates))

# Si hay más de 1200 llamadas en el día, reducir TMO (agentes más rápidos)
tmo = np.where(calls > 1200, tmo * 0.9, tmo)  
tmo = np.round(tmo, 2)  # Redondear a 2 decimales

#Simulacion del Nivel de Servicio 

# Base de nivel de servicio entre 80% y 90%
nivel_servicio = np.random.uniform(80, 90, len(dates))

# Si hay muchas llamadas (>1300), reducir nivel de servicio
nivel_servicio = np.where(calls > 1300, nivel_servicio - 5, nivel_servicio)
nivel_servicio = np.round(nivel_servicio, 2)


#Simulacion de  la Tasa de Ocupación

# Ocupación entre 75% y 90%
ocupacion = np.random.uniform(75, 90, len(dates))

# En días de alto tráfico (>1300 llamadas), aumenta la ocupación
ocupacion = np.where(calls > 1300, ocupacion + 5, ocupacion)
ocupacion = np.round(ocupacion, 2)

#Simulacion del  el Ausentismo

# Base de ausentismo entre 5% y 15%
ausentismo = np.random.uniform(5, 15, len(dates))

# Aumentamos ausentismo los lunes y después de feriados
dias_semana = pd.Series(dates).dt.dayofweek
ausentismo = np.where(dias_semana == 0, ausentismo + 5, ausentismo)  # +5% los lunes
ausentismo = np.round(ausentismo, 2)

#Simulaciom de  la Dotación de Agentes

# Cantidad de agentes programados (entre 50 y 100)
agentes_programados = np.random.randint(50, 100, len(dates))

# Agentes efectivos = programados * (1 - ausentismo)
agentes_efectivos = (agentes_programados * (1 - ausentismo / 100)).astype(int)

#Crear el DataFrame Final

# Creamos el  Df
df = pd.DataFrame({
    'fecha': dates,
    'llamadas': calls,
    'tmo': tmo,
    'nivel_servicio': nivel_servicio,
    'ocupacion': ocupacion,
    'ausentismo': ausentismo,
    'agentes_programados': agentes_programados,
    'agentes_efectivos': agentes_efectivos
})

# Guardamos como CSV
df.to_csv('call_center_data.csv', index=False)

# Muestro  las primeras filas
print(df.head())


#Visualizo los Datos

# Graficar volumen de llamadas
plt.figure(figsize=(12,5))
plt.plot(df['fecha'], df['llamadas'], label='Volumen de llamadas')
plt.xlabel('Fecha')
plt.ylabel('Llamadas')
plt.title('Simulación de Volumen de Llamadas en un Call Center')
plt.legend()
plt.show()
