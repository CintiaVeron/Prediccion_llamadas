import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 📌 Función para entrenar ARIMA y predecir
def forecast_arima(series, steps):
    modelo_arima = sm.tsa.ARIMA(series, order=(5,1,2))
    modelo_entrenado = modelo_arima.fit()
    predicciones = modelo_entrenado.forecast(steps=steps)
    fechas_futuras = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    return pd.DataFrame({'fecha': fechas_futuras, 'llamadas_predichas': predicciones}), modelo_entrenado

# 📌 Función para entrenar Prophet y predecir
def forecast_prophet(df, steps):
    df_prophet = df.reset_index()[['fecha', 'llamadas']]
    df_prophet.columns = ['ds', 'y']
    modelo_prophet = Prophet()
    modelo_prophet.fit(df_prophet)
    future_dates = modelo_prophet.make_future_dataframe(periods=steps)
    forecast = modelo_prophet.predict(future_dates)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'fecha', 'yhat': 'llamadas_predichas'}), modelo_prophet

# 📌 Función para calcular métricas de error
def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 📌 Función para calcular Erlang-C y dotación óptima
def erlang_c(A, N):
    numerador = (A**N / math.factorial(N)) / sum((A**i / math.factorial(i)) for i in range(N + 1))
    denominador = 1 - numerador
    return numerador / denominador if denominador > 0 else 1

def calcular_agentes(volumen_llamadas, tmo, nivel_servicio, ausentismo):
    A = (volumen_llamadas * tmo) / 60  
    N = math.ceil(A)  
    while True:
        E = erlang_c(A, N)  
        probabilidad_atencion = 1 - E  
        if probabilidad_atencion >= nivel_servicio:
            break  
        N += 1  
    return math.ceil(N / (1 - ausentismo))  # Ajuste por ausentismo

# 📌 Cargar datos históricos
df = pd.read_csv("call_center_data.csv")
df['fecha'] = pd.to_datetime(df['fecha'])
df.set_index('fecha', inplace=True)

# 📌 Streamlit UI
st.title("📊 Predicción de Volumen de Llamadas y Dotación de Agentes")

# 🎛️ Parámetros seleccionados por el usuario
dias_prediccion = st.slider("Selecciona el número de días a predecir:", min_value=15, max_value=90, value=30)
tmo = st.number_input("Tiempo Medio de Operación (TMO) en minutos:", min_value=1.0, max_value=15.0, value=6.0)
nivel_servicio = st.slider("Nivel de Servicio (% de llamadas atendidas sin espera):", min_value=0.5, max_value=0.95, value=0.8)
ausentismo = st.slider("Porcentaje de Ausentismo:", min_value=0.0, max_value=0.3, value=0.1)

# 📈 Generar predicciones
predicciones_arima, modelo_arima = forecast_arima(df['llamadas'], dias_prediccion)
predicciones_prophet, modelo_prophet = forecast_prophet(df, dias_prediccion)

# 📊 Gráfico de predicciones
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df['llamadas'], label='Histórico de llamadas', color='blue')
ax.plot(predicciones_arima['fecha'], predicciones_arima['llamadas_predichas'], label='Predicción ARIMA', linestyle='dashed', color='red')
ax.plot(predicciones_prophet['fecha'], predicciones_prophet['llamadas_predichas'], label='Predicción Prophet', linestyle='dotted', color='green')
ax.set_xlabel("Fecha")
ax.set_ylabel("Llamadas")
ax.set_title("📈 Predicción de Volumen de Llamadas")
ax.legend()
st.pyplot(fig)

# 📌 Calcular métricas de error en datos históricos
historico_real = df['llamadas'].iloc[-dias_prediccion:]  # Últimos datos reales
historico_pred_arima = modelo_arima.forecast(steps=dias_prediccion)
df_fechas = df.reset_index()[['fecha']].iloc[-dias_prediccion:].rename(columns={'fecha': 'ds'})
historico_pred_prophet = modelo_prophet.predict(df_fechas)['yhat']


mae_arima, mse_arima, rmse_arima = calcular_metricas(historico_real, historico_pred_arima)
mae_prophet, mse_prophet, rmse_prophet = calcular_metricas(historico_real, historico_pred_prophet)

# 📌 Crear DataFrame con métricas
df_metricas = pd.DataFrame({
    "Modelo": ["ARIMA", "Prophet"],
    "MAE": [mae_arima, mae_prophet],
    "MSE": [mse_arima, mse_prophet],
    "RMSE": [rmse_arima, rmse_prophet]
})

# 📌 Mostrar métricas en Streamlit
st.subheader("📊 Comparación de Modelos")
st.write("📌 **Las métricas de error más bajas indican un mejor modelo.**")
st.dataframe(df_metricas.style.highlight_min(subset=["MAE", "MSE", "RMSE"], color="lightgreen"))

# 📌 Seleccionar mejor modelo basado en RMSE más bajo
mejor_modelo = "ARIMA" if rmse_arima < rmse_prophet else "Prophet"
st.success(f"🏆 **El mejor modelo es: {mejor_modelo}** (Menor RMSE)")

# 📌 Calcular dotación de agentes para el mejor modelo
if mejor_modelo == "ARIMA":
    volumen_llamadas_predicho = predicciones_arima['llamadas_predichas'].iloc[-1]
else:
    volumen_llamadas_predicho = predicciones_prophet['llamadas_predichas'].iloc[-1]

agentes_requeridos = calcular_agentes(volumen_llamadas_predicho, tmo, nivel_servicio, ausentismo)

# 📌 Mostrar resultado de dotación
st.subheader("📞 Volumen de llamadas estimado y dotación de agentes")
st.write(f"🔹 **Volumen de llamadas esperado en {dias_prediccion} días:** {volumen_llamadas_predicho:.0f}")
st.write(f"👩‍💼 **Número de agentes requeridos (ajustado por ausentismo):** {agentes_requeridos}")
