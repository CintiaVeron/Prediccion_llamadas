import pandas as pd
from prophet import Prophet

def entrenar_prophet(df):
    """
    Entrena un modelo Prophet con los datos históricos.
    df: DataFrame con columnas ['fecha', 'llamadas'].
    """
    df = df.rename(columns={'fecha': 'ds', 'llamadas': 'y'})
    modelo = Prophet()
    modelo.fit(df)
    return modelo

def predecir_prophet(modelo, pasos=30):
    """
    Genera predicciones con Prophet.
    modelo: Modelo Prophet entrenado.
    pasos: Número de días a predecir.
    """
    futuro = modelo.make_future_dataframe(periods=pasos)
    prediccion = modelo.predict(futuro)
    return prediccion[['ds', 'yhat']]

