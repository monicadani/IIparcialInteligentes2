from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

np.set_printoptions(suppress=True)

class Prediccion():
    def __init__(self, ruta):
        self.model = load_model(ruta, compile=False)


    def predecir(self, imagen):
        # imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas = []
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA = np.array(imagenesCargadas)
        predicciones = self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=", predicciones)
        clasesMayores = np.argmax(predicciones, axis=1)
        return clasesMayores[0]
