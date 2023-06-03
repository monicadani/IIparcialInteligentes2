"""

@author: Monica y Dalia
"""
import base64
import time
import tensorflow as tf
import keras
import numpy as np
import cv2
from Prediccion import Prediccion


# Se usan las siguientes librerias para trabajo con red Neuronal

from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def cargarDatos(fase, numeroCategorias, limite, width, height):
    imagenesCargadas = []
    valorEsperado = []

    for categoria in range(6, numeroCategorias + 6):
        for idImagen in range(1, limite[categoria - 6] + 1):
            ruta = "../dataset/train/" + str(categoria) + "/" + str(categoria) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width, height))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria - 6] = 1
            valorEsperado.append(probabilidades)
    imagenes_entrenamiento = np.array(imagenesCargadas)
    valores_esperados = np.array(valorEsperado)

    print("CANTIDAD DE IMAGINES", len(imagenes_entrenamiento))
    print("CANTIDAD DE VALORES", len(valores_esperados))

    return imagenes_entrenamiento, valores_esperados


width = 128
height = 128
pixeles = width * height

# Si es a blanco y negro es -> 1 si es RGB es -> 3
num_channels = 1
img_shape = (width, height, num_channels)

# Cant elementos a clasifica
num_clases = 7
cantidad_datos_entenamiento = [50, 50, 50, 50, 50, 50, 50]
cantidad_datos_pruebas = [16, 16, 16, 16, 16, 16,16]

# Carga de los datos
imagenes, probabilidades = cargarDatos(
    "../dataset/train/", num_clases, cantidad_datos_entenamiento, width, height)
print(imagenes)

model = Sequential()

# Capa de entrada
model.add(InputLayer(input_shape=(pixeles,)))

# Re armar la imagen
model.add(Reshape(img_shape))

# Capas convolucionales
model.add(Conv2D(kernel_size=8, strides=2, filters=30,
                 padding="same", activation="selu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=8, strides=2, filters=36,
                 padding="same", activation="selu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))

# Aplanamiento
model.add(Flatten())
model.add(Dense(128, activation="relu"))

# Capa de salida
model.add(Dense(num_clases, activation="softmax"))

# Traducir de keras a tensorflow
model.compile(optimizer="adam",
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=25)
# Pruebas
imagenes_prueba, probabilidades_prueba = cargarDatos(
    "../dataset/test/", num_clases, cantidad_datos_pruebas, width, height)
resultados = model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
print(model.metrics_names)
print(resultados)

# Guardar el modelo
ruta = "models/model_c.h5"
model.save(ruta)

# Estructura de la red

model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)
print("----METRIC RESULT----")
print(metricResult)
start_time = time.time()
scnn_pred = model.predict(imagenes_prueba, batch_size=14, verbose=1)
end_time = time.time()
scnn_predicted = np.argmax(scnn_pred, axis=1)
response_time = end_time - start_time
print("Tiempo de respuesta:", response_time, "segundos")
# Creamos la matriz de confusión
scnn_cm = confusion_matrix(
    np.argmax(probabilidades_prueba, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(7), range(7))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(
    np.argmax(probabilidades_prueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)
