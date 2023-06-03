import cv2
import numpy as np
from Prediccion import Prediccion
from test import test
from Cut import Cut


nameWindow = "Calculadora"
prediccion = Prediccion('models/models/keras_modelPPNuevo.h5')
prueba=test('models/models/models/model_a.h5')


def nothing(x):
    pass


def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 0, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 200000, nothing)


def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas


def showKeys():
    #Borrar luego
    global captura, indexImage, predict, izqVal, derVal, total, acumulado, numA, numB,text

    cv2.putText(imagen, "Presione 'esc' para salir", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Presione 'P' para predecir y realizar la suma de las cartas", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(imagen, "El acomulador es="+str(acumulado), (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(imagen, text, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2, cv2.LINE_AA)



def detectarForma(imagen):
    global captura, indexImage, predict, izqVal, derVal, total, acumulado, numA, numB,text
    cut = Cut()
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gris", imagenGris)

    #
    tamañoKernel = 20
    min = 255
    max = 255
    areaMin = 50000
    #Cambiar area minima---------------------------


    bordes = cv2.Canny(imagenGris, min, max)

    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    # cv2.imshow("Bordes", bordes)
    figuras, jerarquia = cv2.findContours(
        bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)


    bor = []
    i = 0
    showKeys()
    for fig in figuras:

        if areas and areas[i] > areaMin:
            vertices = cv2.approxPolyDP(
                figuras[i], 0.05 * cv2.arcLength(figuras[i], True), True)
            if len(vertices) == 4:
                cv2.drawContours(imagen, [figuras[i]], 0, (0, 0, 255), 2)
                # print(i,"area",areas[i])
                bor.append(i)
                #cv2.putText(imagen, "La suma de " + str(derVal) + " y " + str(izqVal) + " es: " + str(total), (50, 70),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                #cv2.putText(imagen, "El acumulado es: " + str(acumulado), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #            (0, 255, 0), 2, cv2.LINE_AA)

        i = i + 1


    Izq=None;
    Der=None;

    try:

        y = 30
        x = 30
        h = 250
        w = 200

        Izq = cut.crop(imagen, [figuras[bor[0]]], indexImage)
        # Izq = Izq[y:y+h, x:x+w]
        Izq = cv2.resize(Izq, (224, 224), cv2.INTER_AREA)
        cv2.imshow("Izq", Izq)
        cv2.moveWindow("Izq", 40, 30)



        Der = cut.crop(imagen, [figuras[bor[1]]], indexImage)
        # Der = Der[y:y+h, x:x+w]
        Der = cv2.resize(Der, (224, 224), cv2.INTER_AREA)
        cv2.imshow("Der", Der)
        cv2.moveWindow("Der", 180, 30)


    except:
        pass



    if(Izq is not None and Der is not None):
        numA = 5;
        numB = 7;

        print("Datos----------------- ")
        izqVal = prueba.predice_imagen(Izq) + 6
        print("----------------- ")
        derVal = prueba.predice_imagen(Der) + 6

        if predict is True:
            text="CartaA="+str(izqVal)+" CartaB="+str(derVal)
            acumulado=acumulado+izqVal+derVal
            print()




    else:
        if predict is True:
            text="Error, presiona P cuando hayan dos contornos rojo"
            print()





    return imagen


# Apertura cámara
# constructorVentana()
text=""
video = cv2.VideoCapture(0)
bandera = True
predict = False
images64 = []
indexImage = 0
i = 0
acumulado = 0
izqVal = 0
derVal = 0
total = 0

#Valores de testeo
numA=1;
numB=1;


while bandera:
    _, imagen = video.read()
    imagen = detectarForma(imagen)
    imagen = cv2.resize(imagen, (1280, 720), cv2.INTER_AREA)
    cv2.imshow("Imagen", imagen)

    # Parar el programa
    k = cv2.waitKey(5) & 0xFF
    if k == 112:
        predict = True

    if k == 27:  # scape
        bandera = False

    # cv2.imshow("Imagen", imagen)
video.release()
cv2.destroyAllWindows()
