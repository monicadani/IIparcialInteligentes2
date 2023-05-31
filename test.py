
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

np.set_printoptions(suppress=True)

class test():
    def __init__(self, ruta):
        self.model = load_model(ruta, compile=False)

    def predice_imagen(self, imagen):
        # Grab the webcamera's image.
        image =imagen

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = self.model.predict(image)
        index = np.argmax(prediction)
        print("----------")
        print(prediction)
        print(index)

        return index
