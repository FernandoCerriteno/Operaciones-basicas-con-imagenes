import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #Pasar la imagen a escala de grises 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Parametros para la detección de bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) #Detecta los bordes en la imagen en escala de grises utilizando el algoritmo Canny
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #Convierte la imagen de bordes a una imagen en color.
    edges_color[np.where((edges_color == [255, 255, 255]).all(axis=2))] = [0, 255, 255] #Resalta los bordes detectados.

    #Mostrar las imágenes en la pantalla
    cv2.imshow('Escala de Grises', gray)
    cv2.imshow('Bordes resaltados', cv2.addWeighted(frame, 0.8, edges_color, 0.2, 0))

    #Detectar círculos en la imagen
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=0, maxRadius=0)
    #Detectar lineas en la imagen
    lines = cv2.HoughLines(edges, 1, theta=np.pi/180, threshold=100)

    #Mostrar los circulos en frame
    if circles is not None: #Condicional para que no regrese una matriz vacia si no encuentra circulos
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            #Resalta los circulos encontrados en verde
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

    #Mostrar los circulos en frame
    if lines is not None: #Condicional para que no regrese una matriz vacia si no encuentra lineas
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #Resalta las lineas en rojo
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    
    #Binarización de los bordes para una detección mas facil
    ret, thrash = cv2.threshold(edges, 240 , 255, cv2.CHAIN_APPROX_NONE)
    #Encuentra los bordes con la funcion findContours
    contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  

    #Resalta los bordes encontrados en azul
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    #Muestra el video con los poligonos resaltados en la pantalla
    cv2.imshow('Circulos, lineas y poligonos', frame)
    
    #Presionar q para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
