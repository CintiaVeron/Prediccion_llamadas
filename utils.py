import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def cargar_datos(ruta_csv):
    """
    Carga datos desde un CSV.
    ruta_csv: Ruta del archivo CSV.
    """
    return pd.read_csv(ruta_csv)

def calcular_metricas(y_real, y_pred):
    """
    Calcula métricas de error.
    y_real: Valores reales.
    y_pred: Valores predichos.
    """
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
