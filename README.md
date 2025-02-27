# 📊 Mini Proyecto: Predicción de Volumen de Llamadas con ARIMA y Prophet

Este es un **mini proyecto de estudio** en el que exploramos cómo los modelos de series temporales **ARIMA** y **Prophet** se adaptan a la predicción del volumen de llamadas en un call center.  
Además, utilizamos la **fórmula de Erlang-C** para estimar la cantidad óptima de agentes necesarios en función de las predicciones.  

## 📌 Objetivo

El objetivo de este proyecto es comparar los modelos **ARIMA** y **Prophet** en la predicción de llamadas y analizar cómo se comportan en diferentes horizontes de tiempo (30, 60 días, etc.).  
También, a partir de estas predicciones, calcular la dotación de agentes requerida mediante el modelo **Erlang-C**.

## 🛠️ Tecnologías Utilizadas

- **Python** (3.11)
- **Streamlit** (para la interfaz interactiva)
- **Pandas** (manejo de datos)
- **NumPy** (operaciones matemáticas)
- **Matplotlib** (visualización)
- **Statsmodels** (para ARIMA)
- **Prophet** (modelo de predicción de Facebook)
- **Scikit-Learn** (métricas de evaluación)

## 📁 Estructura del Proyecto

```
📂 MiniProyecto_Prediccion_Llamadas
│── 📜 app.py                 # Código principal de la aplicación en Streamlit
│── 📜 modelo_arima.py         # Implementación del modelo ARIMA
│── 📜 modelo_prophet.py       # Implementación del modelo Prophet
│── 📜 erlang_c.py             # Cálculo de dotación de agentes con Erlang-C
│── 📜 utils.py                # Funciones auxiliares
│── 📜 requirements.txt        # Dependencias del proyecto
│── 📜 data_generator.py       # Generador de datos ficticios para pruebas
```

## 📈 Modelos Utilizados

### 🔹 ARIMA (AutoRegressive Integrated Moving Average)

Es un modelo clásico de series temporales que combina:

- **AR (Autoregressive)**: Dependencia entre valores pasados.
- **I (Integrated)**: Diferenciación para hacer la serie estacionaria.
- **MA (Moving Average)**: Promedia errores pasados.

**Ejemplo de configuración:**  
`order=(5,1,2)` → 5 términos AR, 1 diferenciación, 2 términos MA.

---

### 🔹 Prophet

Modelo desarrollado por **Facebook**, basado en:

- **Tendencias** (cambios en el tiempo).
- **Estacionalidad** (diaria, semanal, anual).
- **Efectos de vacaciones o eventos especiales**.

**Ventaja:** Funciona bien con datos irregulares y con estacionalidad fuerte.

---

### 🔹 Erlang-C (Cálculo de Dotación de Agentes)

Usamos **Erlang-C** para estimar la cantidad óptima de agentes según el volumen de llamadas.  
La fórmula es:

\(
P(W>0) = rac{rac{(A^N / N!)}{\sum_{i=0}^{N} (A^i / i!)}}{1 - \left( rac{(A^N / N!)}{\sum_{i=0}^{N} (A^i / i!)} ight)}
\)

Donde:

- **A** = Carga de trabajo en Erlangs  
- **N** = Número de agentes  

El resultado indica la **probabilidad de que una llamada espere en cola**.  
Si el nivel de servicio no es suficiente, aumentamos **N** hasta cumplir el objetivo.

## 📊 Métricas de Evaluación

Para comparar los modelos usamos:

- **MAE (Error Absoluto Medio):** \( MAE = \frac{1}{n} \sum |y_i - \hat{y_i}| \)  
  → Mide el error promedio en unidades reales.

- **MSE (Error Cuadrático Medio):** \( MSE = \frac{1}{n} \sum (y_i - \hat{y_i})^2 \)  
  → Penaliza más los errores grandes.

- **RMSE (Raíz del Error Cuadrático Medio):** \( RMSE = \sqrt{MSE} \)  
  → Similar a MSE, pero en las mismas unidades que la variable original.

### 🏆 ¿Cuál modelo es mejor?

- **A corto plazo** (hasta 30 días) → ARIMA suele ser más preciso.  
- **A largo plazo** (más de 30 días) → Prophet se adapta mejor a tendencias y estacionalidad.  

## 🚀 Cómo Ejecutar

1. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación en Streamlit**  
   ```bash
   streamlit run app.py
   ```

3. **Explorar predicciones y ajustar parámetros**  

## 📌 Notas Finales

- **Este es un proyecto de estudio** con datos ficticios.  
- Se puede mejorar ajustando hiperparámetros y probando más horizontes de predicción.  

---

📩 _Si te interesa el tema, ¡colabora con mejoras o ideas!_ 🚀
