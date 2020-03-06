#INCOMPLETO
#TERMINAR ESTE MODULO CUANDO TENGA MAS FOTOS EN MI BASE DE DATOS MAS GRANDE


import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from skimage import io, color, img_as_float, img_as_ubyte, filters

import cv2

def img2gray(image, mode='sk'):
    if (mode=='sk'):
        gray = color.rgb2gray(image)
    elif (mode=='cv'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def normSize(image, size=(tuple((400, 300)))):
    image = cv2.resize(image, size)
    return image

def imgClean(image, sigma=1, mode='sk'):
    if (mode == 'sk'):
        clean = filters.gaussian(image, sigma)
    elif (mode == 'cv'):
        clean = cv2.GaussianBlur(img, (3, 3), 0)
    return clean

def imgEdge(image, mode='sk'):
    if (mode == 'sk'):
        edge = filters.sobel(image)
    elif (mode == 'cv'):
        edge = cv2.Laplacian(image, cv2.CV_64F)
    return edge

def threshold(image, mode='sk'):
    if (mode == 'sk'):
        th = filters.threshold_isodata(image)
    elif (mode == 'cv'):
        ret, th = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return (image < th)

#Determinacion de la base de datos
tornillo = io.ImageCollection('./data/tornillos/*.png:./data/tornillos/*.jpg')
tuerca = io.ImageCollection('./data/tuercas/*.png:./data/tuercas/*.jpg')
arandela = io.ImageCollection('./data/arandelas/*.png:./data/arandelas/*.jpg')
clavo = io.ImageCollection('./data/clavos/*.png:./data/clavos/*.jpg')

tornillo_gray = []
tornillo_n = []
tornillo_edge = []

tuerca_gray = []
tuerca_n = []
tuerca_edge = []

arandela_gray = []
arandela_n = []
arandela_edge = []

clavo_gray = []
clavo_n = []
clavo_edge = []

i = 0
for i in range(0, len(tornillo)-1):
    aux = normSize(tornillo[i])
    # aux = imgClean(aux, mode='cv')
    tornillo_n.append(aux)
    tornillo_gray.append(img2gray(tornillo_n[i], mode='cv'))
    tornillo_edge.append(imgEdge(tornillo_gray[i]))

i = 0
for i in range(0, len(tuerca)-1):
    aux = normSize(tuerca[i])
    # aux = imgClean(aux, mode='cv')
    tuerca_n.append(aux)
    tuerca_gray.append(img2gray(tuerca_n[i], mode='cv'))
    tuerca_edge.append(imgEdge(tuerca_gray[i]))

i = 0
for i in range(0, len(arandela)-1):
    aux = normSize(arandela[i])
    # aux = imgClean(aux, mode='cv')
    arandela_n.append(aux)
    arandela_gray.append(img2gray(arandela_n[i], mode='cv'))
    arandela_edge.append(imgEdge(arandela_gray[i]))
    
i = 0
for i in range(0, len(clavo)-1):
    aux = normSize(clavo[i])
    # aux = imgClean(aux, mode='cv')
    clavo_n.append(aux)
    clavo_gray.append(img2gray(lemon_n[i], mode='cv'))
    clavo_edge.append(imgEdge(lemon_gray[i]))
    
