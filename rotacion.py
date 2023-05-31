import cv2
import numpy as np

def procesar_imagen(image_path,titulo):
    # Cargar la imagen
    img = cv2.imread(image_path)


    # Rotar la imagen recortada en diferentes ángulos
    for angle in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -130]:

        # Rotar la imagen recortada
        rotated = rotar_imagen(img, angle)

        # Convertir la imagen rotada a escala de grises
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # Redimensionar la imagen rotada a 128x128 píxeles
        resized = cv2.resize(rotated_gray, (128, 128))

        # Guardar la imagen redimensionada para el contorno actual
        cv2.imwrite(f"{titulo}rotated_{angle}.jpg", resized)

def rotar_imagen(image, angle):
    # Obtener la matriz de rotación
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)

    # Aplicar la rotación a la imagen
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return rotated

