import cv2
import numpy as np

def recortar_contorno(image, contour):
    # Crear una imagen en blanco del mismo tamaño que la imagen original
    contour_image = np.zeros_like(image)

    # Dibujar el contorno en la imagen en blanco
    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)

    # Recortar la región dentro del contorno en la imagen original
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]

    return cropped


def rotar_imagen(image, angle):
    # Obtener la matriz de rotación
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)

    # Aplicar la rotación a la imagen
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return rotated


def procesar_imagen(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre cada contorno y crear una imagen para cada uno
    for i, contour in enumerate(contours):
        # Recortar el contorno de la imagen
        cropped = recortar_contorno(img, contour)

        # Rotar la imagen recortada en diferentes ángulos
        for angle in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -130]:

            # Rotar la imagen recortada
            rotated = rotar_imagen(cropped, angle)

            # Convertir la imagen rotada a escala de grises
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

            # Redimensionar la imagen rotada a 128x128 píxeles
            resized = cv2.resize(rotated_gray, (128, 128))

            # Guardar la imagen redimensionada para el contorno actual
            cv2.imwrite(f"contorno_{i}_rotated_{angle}.jpg", resized)


procesar_imagen("6.0.jpg")


