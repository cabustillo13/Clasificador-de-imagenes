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
    clavo_gray.append(img2gray(clavo_n[i], mode='cv'))
    clavo_edge.append(imgEdge(clavo_gray[i]))

#HOG 
from skimage.feature import hog

def m_hog(image):
    feature = hog(image, block_norm='L2-Hys').ravel()
    return feature

#Elementos estadisticos
def stats(arr):
    
    sum = 0
    for value in arr:
        sum += value
    med = sum / len(arr)
    sum = 0
    for value in arr:
        sum += np.power((value - med), 2)
    dstd = np.sqrt(sum / (len(arr) - 1))
    
    return med, dstd

#f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (15, 5))
fig_hog, ax = plt.subplots()

#HOG- Reduccion de dimension
import matplotlib.patches as mpatches

for objeto in tornillo_gray:
    tornillo_fhog = m_hog(objeto)
    med, dstd = stats(tornillo_fhog)
    ax.plot(med, dstd, 'o', color='yellow')
    
for objeto in tuerca_gray:
    tuerca_fhog = m_hog(objeto)
    med, dstd = stats(tuerca_fhog)
    ax.plot(med, dstd, 'o', color='red')
    
for objeto in arandela_gray:
    arandela_fhog = m_hog(objeto)
    med, dstd = stats(arandela_fhog)
    ax.plot(med, dstd, 'o', color='blue')

for objeto in clavo_gray:
    clavo_fhog = m_hog(objeto)
    med, dstd = stats(clavo_fhog)
    ax.plot(med, dstd, 'o', color='green')

ax.grid(True)
ax.set_title("HOG - Reduccion de dimension")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')

plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion Estandar')
plt.xlabel('Media Aritmetica')
plt.show()

#Momentos de Hu- Reduccion de dimension
def hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

fig_hm, ax = plt.subplots()

for objeto in tornillo_edge:
    tornillo_fhm = hu_moments(objeto)
    med, dstd = stats(tornillo_fhm)
    ax.plot(med, dstd, 'o', color='yellow')
    
for objeto in tuerca_edge:
    tuerca_fhm = hu_moments(objeto)
    med, dstd = stats(tuerca_fhm)
    ax.plot(med, dstd, 'o', color='red')
    
for objeto in arandela_edge:
    arandela_fhm = hu_moments(objeto)
    med, dstd = stats(arandela_fhm)
    ax.plot(med, dstd, 'o', color='blue')

for objeto in clavo_edge:
    clavo_fhm = hu_moments(objeto)
    med, dstd = stats(clavo_fhm)
    ax.plot(med, dstd, 'o', color='green')

ax.grid(True)
ax.set_title("Momentos de Hu - Reduccion de Dimension")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label = 'Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion Estandar')
plt.xlabel('Media Aritmetica')
plt.show()

#Haralick
import mahotas

def haralick(image):
    feature = mahotas.features.haralick(image).mean(axis=0)
    return feature

fig_ht, ax = plt.subplots()

for objeto in tornillo_gray:
    tornillo_fht = haralick(objeto)
    med, dstd = stats(tornillo_fht)
    ax.plot(med, dstd, 'o', color='yellow')
    
for objeto in tuerca_gray:
    tuerca_fht = haralick(objeto)
    med, dstd = stats(tuerca_fht)
    ax.plot(med, dstd, 'o', color='red')
    
for objeto in arandela_gray:
    arandela_fht = haralick(objeto)
    med, dstd = stats(arandela_fht)
    ax.plot(med, dstd, 'o', color='blue')

for objeto in clavo_gray:
    clavo_fht = haralick(objeto)
    med, dstd = stats(clavo_fht)
    ax.plot(med, dstd, 'o', color='green')

ax.grid(True)
ax.set_title("Haralick - Reduccion de Dimension")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label = 'Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion Estandar')
plt.xlabel('Media Aritmetica')
plt.show()
