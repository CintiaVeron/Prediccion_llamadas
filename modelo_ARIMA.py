import pandas as pd
import statsmodels.api as sm

def entrenar_arima(df, orden=(5,1,0)):
    """
    Entrena un modelo ARIMA con los datos históricos.
    df: DataFrame con la serie temporal.
    orden: Parámetros del modelo ARIMA (p, d, q).
    """
    modelo = sm.tsa.ARIMA(df['llamadas'], order=orden)
    modelo_entrenado = modelo.fit()
    return modelo_entrenado

def predecir_arima(modelo, pasos=30):
    """
    Genera predicciones con el modelo ARIMA.
    modelo: Modelo ARIMA entrenado.
    pasos: Número de días a predecir.
    """
    predicciones = modelo.forecast(steps=pasos)
    return predicciones

