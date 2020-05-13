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

def normSize(image, size=(tuple((500, 400)))):
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
tornillo = io.ImageCollection('./Data Base/YTrain/ZTornillos/*.png:./Data Base/YTrain/ZTornillos/*.jpg')
tuerca = io.ImageCollection('./Data Base/YTrain/ZTuercas/*.png:./Data Base/YTrain/ZTuercas/*.jpg')
arandela = io.ImageCollection('./Data Base/YTrain/ZArandelas/*.png:./Data Base/YTrain/ZArandelas/*.jpg')
clavo = io.ImageCollection('./Data Base/YTrain/ZClavos/*.png:./Data Base/YTrain/ZClavos/*.jpg')

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
    clavo_gray.append(img2gray(clavo_n[i], mode='cv'))
    clavo_edge.append(imgEdge(clavo_gray[i]))
    
#HOG: Histograma de gradientes orientados
from skimage.feature import hog

def m_hog(image):
    feature = hog(image, block_norm='L2-Hys').ravel()
    return feature

##3 tornillos distintos
t1_fhog = m_hog(tornillo_gray[0])
t2_fhog = m_hog(tornillo_gray[1])
t3_fhog = m_hog(tornillo_gray[2])

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

ax0.plot(t1_fhog, color='C3')
ax1.plot(t2_fhog, color='C3')
ax2.plot(t3_fhog, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
plt.show()

##Entre tornillo, tuerca, arandela y clavo
tornillo_fhog = m_hog(tornillo_gray[0])
clavo_fhog = m_hog(clavo_gray[0])
tuerca_fhog = m_hog(tuerca_gray[0])
arandela_fhog = m_hog(arandela_gray[0])

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15, 5))

ax0.plot(tornillo_fhog, color='C3')
ax1.plot(tuerca_fhog, color='C3')
ax2.plot(arandela_fhog, color='C3')
ax3.plot(clavo_fhog, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
plt.show()

#Momentos de Hu

##Entre 3 tornillos distintos
def hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

t1_fhm = hu_moments(tornillo_edge[0])
t2_fhm = hu_moments(tornillo_edge[1])
t3_fhm = hu_moments(tornillo_edge[2])

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

ax0.plot(t1_fhm, color='C3')
ax1.plot(t2_fhm, color='C3')
ax2.plot(t3_fhm, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
plt.show()

##Entre tornillo, tuerca, arandela y clavo
tornillo_fhm = hu_moments(tornillo_edge[0])
tuerca_fhm = hu_moments(tuerca_edge[0])
arandela_fhm = hu_moments(arandela_edge[0])
clavo_fhm = hu_moments(clavo_edge[0])

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15, 5))

ax0.plot(tornillo_fhm, color='C3')
ax1.plot(tuerca_fhm, color='C3')
ax2.plot(arandela_fhm, color='C3')
ax3.plot(clavo_fhm, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
plt.show()

#Haralick Textura
import mahotas

def haralick(image):
    feature = mahotas.features.haralick(image).mean(axis=0)
    return feature

##Entre 3 clavos distintos
t1_fht = haralick(tornillo_gray[0])
t2_fht = haralick(tornillo_gray[1])
t3_fht = haralick(tornillo_gray[2])

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

ax0.plot(t1_fht, color='C3')
ax1.plot(t2_fht, color='C3')
ax2.plot(t3_fht, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
plt.show()

##Entre tornillo, tuerca, arandela y clavo
tornillo_fht = haralick(tornillo_gray[0])
clavo_fht = haralick(clavo_gray[0])
tuerca_fht = haralick(tuerca_gray[0])
arandela_fht = haralick(arandela_gray[0])

f, (ax0, ax2, ax3, ax1) = plt.subplots(1, 4, figsize=(15, 5))

ax0.plot(tornillo_fht, color='C3')
ax1.plot(clavo_fht, color='C3')
ax2.plot(tuerca_fht, color='C3')
ax3.plot(arandela_fht, color='C3')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
plt.show()
