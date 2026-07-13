#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import os
import gzip
import pickle as pk
import json
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

def lectura_limpieza(ruta):
    data=pd.read_csv(ruta, index_col=False, compression="zip")
    data["Age"]=2021-data["Year"]
    data.drop(columns=["Year","Car_Name"], inplace=True)
    return data

def division(data):
    return data.drop(columns=["Present_Price"]),data["Present_Price"]

def crear_pipeline():
    variables_categoricas=["Fuel_Type","Selling_type","Transmission","Owner"]
    variables_numericas=["Selling_Price","Driven_kms","Age"]
    preprocessing= ColumnTransformer(
        transformers=[
            (
                "One",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categoricas
            ),
            (
                "escalado",
                MinMaxScaler(),
                variables_numericas
            )
        ],
        remainder="passthrough",
    )
    pipeline=Pipeline(
        steps=[
            ("preprocesado",preprocessing),
            ("SelectKBest", SelectKBest(score_func=f_regression)),
            ("modelo", LinearRegression())
        ]
    )
    return pipeline

def optimizador(x_train,y_train):
    parametros = {
        "SelectKBest__k": list(range(1, 20)),
  
    }

    modelo = GridSearchCV(
        estimator=crear_pipeline(),
        param_grid=parametros,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    modelo.fit(x_train, y_train)

    return modelo

def guardar_modelo(modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz","wb") as archivo:
        pk.dump(modelo,archivo)

def calcular_metricas(nombre, reales, predichos):
    return {
        "type": "metrics",
        "dataset": nombre,
        "r2": round(r2_score(reales, predichos), 4),
        "mse": round(mean_squared_error(reales, predichos), 4),
        "mad": round(mean_absolute_error(reales, predichos), 4),
    }
    
def guardar_metricas(resultados):
    os.makedirs("files/output", exist_ok=True)

    with open("files/output/metrics.json","w",) as archivo:
        for resultado in resultados:
            archivo.write(json.dumps(resultado))
            archivo.write("\n")
            
def main():
    train=lectura_limpieza("files/input/train_data.csv.zip")
    test=lectura_limpieza("files/input/test_data.csv.zip")
    x_train, y_train=division(train)
    x_test, y_test=division(test)
    modelo=optimizador(x_train,y_train)
    guardar_modelo(modelo)
    pred_train=modelo.predict(x_train)
    pred_test=modelo.predict(x_test)

    resultados = [
        calcular_metricas(
            "train",
            y_train,
            pred_train,
        ),
        calcular_metricas(
            "test",
            y_test,
            pred_test,
        ),
    ]

    guardar_metricas(resultados)
if __name__=="__main__":
    main()