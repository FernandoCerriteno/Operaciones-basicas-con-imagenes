import cv2
import numpy as np

imageList = ['road0.png', 'road10.png', 'road11.png', 'road12.png', 'road108.png', 'road114.png', 'road115.png', 'road116.png', 'road121.png', 'road122.png']
for i in imageList:
    imagen = cv2.imread("D:/fcmdr/Documents/Programas/Python/Semestre6/ImageDet/"+i)
    cv2.imshow("Imagen a color", imagen)
    escalaGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagen en escala de grises", escalaGris)

    umbral, imagen_binaria = cv2.threshold(escalaGris, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("Imagen binaria con umbral = 125", imagen_binaria)

    azul, verde, rojo = cv2.split(imagen)
    umbral, imagen_ColorBin = cv2.threshold(rojo, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("Imagen binaria con umbral = 125 en el color rojo", imagen_ColorBin)

    cv2.waitKey(0)

cv2.destroyAllWindows()

for i in imageList:
    imagen = cv2.imread("D:/fcmdr/Documents/Programas/Python/Semestre6/ImageDet/"+i)
    escalaGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagen en escala de grises", escalaGris)

    ruido = np.zeros(escalaGris.shape, dtype=np.uint8)
    cv2.randn(ruido, 0, 10)
    imagen_Ruido10 = cv2.add(escalaGris, ruido)

    ruido = np.zeros(escalaGris.shape, dtype=np.uint8)
    cv2.randn(ruido, 0, 20)
    imagen_Ruido20 = cv2.add(escalaGris, ruido)

    ruido = np.zeros(escalaGris.shape, dtype=np.uint8)
    cv2.randn(ruido, 0, 30)
    imagen_Ruido30 = cv2.add(escalaGris, ruido)

    cv2.imshow("Imagen en escala de grises", escalaGris)
    cv2.imshow("Imagen con ruido nivel 10", imagen_Ruido10)
    cv2.imshow("Imagen con ruido nivel 20", imagen_Ruido20)
    cv2.imshow("Imagen con ruido nivel 30", imagen_Ruido30)

    kernel = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

    # Aplicar filtro binomial a la imagen
    imagen_filtrada10 = cv2.filter2D(imagen_Ruido10, -1, kernel)
    imagen_filtrada20 = cv2.filter2D(imagen_Ruido20, -1, kernel)
    imagen_filtrada30 = cv2.filter2D(imagen_Ruido30, -1, kernel)

    cv2.imshow("Imagen con filtro binomial en ruido nivel 10", imagen_filtrada10)
    cv2.imshow("Imagen con filtro binomial en ruido nivel 20", imagen_filtrada20)
    cv2.imshow("Imagen con filtro binomial en ruido nivel 30", imagen_filtrada30)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Avarage filter
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    Fil_img = cv2.filter2D( imagen_Ruido10,-1,kernel)
    Fil_img2 = cv2.filter2D(imagen_Ruido20,-1,kernel)
    Fil_img3 = cv2.filter2D(imagen_Ruido30,-1,kernel)

    cv2.imshow('Imagen con filtro aplicado ruido 10', Fil_img)
    cv2.imshow('Imagen con filtro aplicado ruido 20', Fil_img2)
    cv2.imshow('Imagen con filtro aplicado ruido 30', Fil_img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Filtro Gausiano
    FGaus = cv2.GaussianBlur(imagen_Ruido10, (5,5), 0)
    FGaus2 = cv2.GaussianBlur(imagen_Ruido20, (5,5), 0)
    FGaus3 = cv2.GaussianBlur(imagen_Ruido30, (5,5), 0)
    cv2.imshow('Imagen con filtro gausiano aplicado ruido 10', FGaus)
    cv2.imshow('Imagen con filtro gausiano aplicado ruido 20', FGaus2)
    cv2.imshow('Imagen con filtro gausiano aplicado ruido 30', FGaus3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Filtro mediana
    FMed = cv2.medianBlur(imagen_Ruido10, 5)
    FMed2 = cv2.medianBlur(imagen_Ruido20, 5)
    FMed3 = cv2.medianBlur(imagen_Ruido30, 5)
    cv2.imshow('Imagen con filtro median aplicado ruido 10', FMed)
    cv2.imshow('Imagen con filtro median aplicado ruido 20', FMed2)
    cv2.imshow('Imagen con filtro median aplicado ruido 30', FMed3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.destroyAllWindows()