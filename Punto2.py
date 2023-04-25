import numpy as np
import cv2

Edificios = ['Edificio1.jpg','Edificio2.jpg','Edificio3.jpg','Edificio4.jpg','Edificio5.jpg']

for i in Edificios:
    Ecolor = cv2.imread("D:/fcmdr/Documents/Programas/Python/Semestre6/ImageDet/" + i)
    Egray = cv2.cvtColor(Ecolor,cv2.COLOR_BGR2GRAY)
    HistGray = Egray;

    cv2.imshow('Imagen de edificio a color', Ecolor)
    canny = cv2.GaussianBlur(Egray, (3, 3), 0)
    ejes = cv2.Canny(canny, 100, 200, apertureSize=5)
    cv2.imshow('Detector de bordes por canny', ejes)
    cv2.waitKey(0)

    #Operaciones morfologicas
    Kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #Erocion
    c = cv2.morphologyEx(ejes, cv2.MORPH_CLOSE,Kernel)
    ero = cv2.erode(c,Kernel,iterations=1)
    cv2.imshow('Operacion de erosion',ero)
    cv2.waitKey(0)
    #Dilatacion
    Dilate = cv2.dilate(ejes, Kernel, iterations=1)
    cv2.imshow('Operacion de dilatacion',Dilate)   
    cv2.waitKey(0)
    #Apertura
    cerr = cv2.morphologyEx(ejes,cv2.MORPH_CLOSE,Kernel)
    open = cv2.morphologyEx(cerr,cv2.MORPH_OPEN,Kernel)
    cv2.imshow('Operacion de apertura',open)    
    cv2.waitKey(0)
    #Cierre
    close = cv2.morphologyEx(ejes, cv2.MORPH_CLOSE, Kernel)
    cv2.imshow('Operacion de cierre',close)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

    #Histeresis
    His = cv2.GaussianBlur(HistGray, (3,3), 0)
    sobelx = cv2.Sobel(His, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(His, cv2.CV_64F, 0, 1, ksize=3)  
    mag, ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)  
    low_lim = 50
    up_lim= 150
    ejesH = np.zeros_like(Ecolor)
    ejesH[(mag >= low_lim) & (mag <= up_lim)] = 255
    ejesH = cv2.dilate(ejesH, None)
    ejesH = cv2.erode(ejesH, None)
    
    cv2.imshow('Detector de bordes por Hysteresis Thresholding',ejesH)
    cv2.waitKey(0)

    #Operaciones morfologicas
    Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #Erosion
    Erode = cv2.erode(ejesH,Kernel, iterations=1)
    cv2.imshow('Operacion de erosion',Erode)
    cv2.waitKey(0)
    #Dilatacion
    Dilate = cv2.dilate(ejesH,Kernel, iterations=1)
    cv2.imshow('Operacion de dilatacion',Dilate)  
    cv2.waitKey(0) 
    #Apertura
    open = cv2.morphologyEx(ejesH, cv2.MORPH_OPEN, Kernel)
    cv2.imshow('Operacion de apertura',open)    
    cv2.waitKey(0)
    #cierre
    close = cv2.morphologyEx(ejesH, cv2.MORPH_CLOSE, Kernel)
    cv2.imshow('Operacion de cierre',close)         
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()