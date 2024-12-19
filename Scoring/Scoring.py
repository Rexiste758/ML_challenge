# #### Importamos las librerias necesarias

import joblib
import os
import mysql.connector
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Cargamos el modelo entrenado en "Training.py" usando la función joblib.load() y el transformador PolynomialFeatures
# 
# Usamos PolynomialFeatures para transformar un conjunto de características numéricas en un conjunto de características polinomiales.


# Cargamos el modelo en el notebook
poly_reg_model = joblib.load('modelo_entrenado.pkl')


# Cargamos el transformador PolynomialFeatures
poly = joblib.load('transformador_polynomial.pkl')


# Obtener las variables de entorno
db_host = os.getenv('DB_HOST', 'localhost')
db_user = os.getenv('DB_USER', 'myuser')
db_password = os.getenv('DB_PASSWORD', 'mypassword')
db_name = os.getenv('DB_NAME', 'mydatabase')


# #### Realizamos la conexión a la base de datos de MySQL, previamente creada (usando el archivo csv compartido). Al mismo tiempo que, creamos una excepción en caso de que se llegué a generar un mensaje de error.

# Conectar a la base de datos MySQL

connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

print("Conectado a la base de datos MySQL")


# #### Una vez realizada la conexión, obtenemos todo el contenido de la tabla mediente una sentencia de SQL y almacenamos el contenido en la variable df.

df = pd.read_sql_query("select * from training_dataset", connection)


df['log_charges'] = np.log(df['charges'] )


# Realizamos la transformación de la variable "charges" aplicando la función de logartimo y almacenando los nuevos valores en una  columna llamada "log_charges", esto basado en la distribución de la variable y las diferencias de magnitudes con las demás variables.


# Definimos el orden de las categorías para las columnas
encoding_orders = {
    'sex': ['male', 'female'],
    'smoker': ['no', 'yes'],
    'region': ['southeast','southwest','northwest','northeast']
}

# Aplicamos la codificación ordinal para cada columna
for column, order in encoding_orders.items():
    df[column] = df[column].map({cat: idx for idx, cat in enumerate(order)})


# En el script Training.py ya identificamos las frecuencias para cada columna categorica por lo que, es facil definir los valores que puede tomar cada columna.


# Seleccionamos 10 valores aleatorios de la tabla
muestra_aleatoria = df.sample(n=10, random_state=123)

print(muestra_aleatoria)


# Una vez que, hemos realizado las transformaciones y codificaciones necesarias al dataframe (df) para que la data este lista para ser ingresada al modelo. Tomamos de manera aleatoria 10 registros de la tabla usando la funcion sample(), además de plantar la semilla para la reproductividad usando "random_state=123".


# Nos aseguramos de que la muestra aleatoria tiene las mismas columnas de entrada
X_unseen = muestra_aleatoria[["smoker","children","age",'region',"bmi","sex"]]
y_unseen = muestra_aleatoria["log_charges"]  

# Transformamos las características con PolynomialFeatures
poly_features_unseen = poly.transform(X_unseen)

# Realizamos predicciones con el modelo entrenado
poly_y_unseen_predict = poly_reg_model.predict(poly_features_unseen)

# Calculamos el error RMSE para la muestra no vista
poly_rmse_unseen = np.sqrt(mean_squared_error(y_unseen, poly_y_unseen_predict))
print(f"poly RMSE en muestra no vista = {poly_rmse_unseen}")

# Mostramos las predicciones y los valores reales
muestra_aleatoria["predicted_log_charges"] = poly_y_unseen_predict


# ##### 1.- Selección de las columnas de entrada y salida:
# 
# X_unseen = muestra_aleatoria[["smoker","children","age",'region',"bmi","sex"]] \
# y_unseen = muestra_aleatoria["log_charges"] \
# Se extraen las columnas de entrada (X_unseen) y la variable objetivo (y_unseen) de la muestra aleatoria.
# Esto asegura que los datos utilizados tengan las mismas características que el modelo espera.
# 
# 
# ##### 2.- Transformación de características:
# poly_features_unseen = poly.transform(X_unseen) \
# Se aplica la misma transformación "PolynomialFeatures" que se utilizó en el conjunto de entrenamiento.
# "poly.transform" transforma X_unseen para que incluya las características polinómicas necesarias.
# 
# ##### 3.- Predicción con el modelo entrenado:
# 
# poly_y_unseen_predict = poly_reg_model.predict(poly_features_unseen) \
# El modelo entrenado (poly_reg_model) realiza predicciones sobre las características polinómicas de la muestra no vista.
# Los resultados se almacenan en poly_y_unseen_predict.
# 
# ##### 4.- Cálculo del RMSE en la muestra no vista:
# 
# poly_rmse_unseen = np.sqrt(mean_squared_error(y_unseen, poly_y_unseen_predict)) \
# print(f"poly RMSE en muestra no vista = {poly_rmse_unseen}") 
# 
# Se calcula el RMSE (raíz del error cuadrático medio), que mide qué tan lejos están las predicciones (poly_y_unseen_predict) de los valores reales (y_unseen). El RMSE obtenido es 0.15800897766227984, lo cual sugiere que el error en la muestra no vista es bajo y el modelo generaliza bien.
# 
# ##### 5.-Agregar las predicciones a la muestra:
# 
# muestra_aleatoria["predicted_log_charges"] = poly_y_unseen_predict \
# Se añade una nueva columna predicted_log_charges a la muestra aleatoria, donde se almacenan las predicciones del modelo.
# 
# 
# ##### Conclusión:
# 
# ##### poly RMSE en muestra no vista = 0.1580:
# 
# El RMSE en la muestra no vista es bajo (0.15800897766227984), lo que indica que el modelo sigue siendo preciso al predecir datos nuevos no utilizados en el entrenamiento. Valores bajos de RMSE significan que las predicciones del modelo están muy cercanas a los valores reales.


# Aplicamos la exponencial a la columna de predicciones
muestra_aleatoria["predicted_charges"] = np.exp(muestra_aleatoria["predicted_log_charges"])

# Mostramos la tabla con los valores reales, predicciones logarítmicas y predicciones exponenciales
print(muestra_aleatoria[["charges","log_charges", "predicted_log_charges", "predicted_charges"]])


# ##### Aplicamos la función exponencial a las predicciones logarítmicas:
# 
# muestra_aleatoria["predicted_charges"] = np.exp(muestra_aleatoria["predicted_log_charges"])
# 
# np.exp() aplica la función exponencial a los valores de la columna predicted_log_charges.
# Como el modelo hizo predicciones sobre la variable logarítmica (log_charges), se necesita aplicar la exponencial para regresar las predicciones a su escala original (charges).
# 
# 
# ##### Comparación entre charges y predicted_charges:
# 
# Los valores de predicted_charges son generalmente cercanos a los valores reales (charges), pero hay diferencias en algunos casos.
# 
# ###### Comportamiento del modelo:
# 
# El modelo predice bien para valores pequeños y medianos, pero presenta diferencias en valores extremos.
# 
# ##### Importancia de la transformación:
# 
# La transformación logarítmica permitió capturar mejor la estructura de los datos y reducir el efecto de valores muy grandes.
# Al aplicar la exponencial, regresamos las predicciones a su escala original para que puedan compararse directamente con los valores reales.


# Visualizamos la diferencia entre los precios actuales y los que el modelo logró predecir
plt.scatter(muestra_aleatoria["log_charges"],muestra_aleatoria["predicted_log_charges"], color='#029386')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Valores calculados')
plt.grid(axis='y', alpha=0.5)
plt.show()


r2 = r2_score(muestra_aleatoria["log_charges"],muestra_aleatoria["predicted_log_charges"])
print(f"R-squared: {r2}")


# ##### R^2 alto (0.9572): El modelo captura muy bien la tendencia de los datos no vistos.
# 
# Los resultados muestran que el modelo tiene un buen desempeño en general, pero puede subestimar valores extremos.
