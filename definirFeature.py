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
tornillo = io.ImageCollection('./Hog, Haralick y Hu/Cortadas/Tornillos/*.png:./Hog, Haralick y Hu/Cortadas/Tornillos/*.jpg')
tuerca = io.ImageCollection('./Hog, Haralick y Hu/Cortadas/Tuercas/*.png:./Hog, Haralick y Hu/Cortadas/Tuercas/*.jpg')
arandela = io.ImageCollection('./Hog, Haralick y Hu/Cortadas/Arandelas/*.png:./Hog, Haralick y Hu/Cortadas/Arandelas/*.jpg')
clavo = io.ImageCollection('./Hog, Haralick y Hu/Cortadas/Clavos/*.png:./Hog, Haralick y Hu/Cortadas/Clavos/*.jpg')

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
for i in range(0, len(tornillo)):
    aux = normSize(tornillo[i])
    # aux = imgClean(aux, mode='cv')
    tornillo_n.append(aux)
    tornillo_gray.append(img2gray(tornillo_n[i], mode='cv'))
    tornillo_edge.append(imgEdge(tornillo_gray[i]))

i = 0
for i in range(0, len(tuerca)):
    aux = normSize(tuerca[i])
    # aux = imgClean(aux, mode='cv')
    tuerca_n.append(aux)
    tuerca_gray.append(img2gray(tuerca_n[i], mode='cv'))
    tuerca_edge.append(imgEdge(tuerca_gray[i]))

i = 0
for i in range(0, len(arandela)):
    aux = normSize(arandela[i])
    # aux = imgClean(aux, mode='cv')
    arandela_n.append(aux)
    arandela_gray.append(img2gray(arandela_n[i], mode='cv'))
    arandela_edge.append(imgEdge(arandela_gray[i]))
    
i = 0
for i in range(0, len(clavo)):
    aux = normSize(clavo[i])
    # aux = imgClean(aux, mode='cv')
    clavo_n.append(aux)
    clavo_gray.append(img2gray(clavo_n[i], mode='cv'))
    clavo_edge.append(imgEdge(clavo_gray[i]))
    
#HOG: Histograma de gradientes orientados
#from skimage.feature import hog

#def m_hog(image):
#    feature = hog(image, block_norm='L2-Hys').ravel()
#    return feature

##3 tornillos distintos
#print("Entre 3 tornillos")
#t1_fhog = m_hog(tornillo_gray[0])
#print(t1_fhog)
#t2_fhog = m_hog(tornillo_gray[1])
#print(t2_fhog)
#t3_fhog = m_hog(tornillo_gray[2])
#print(t3_fhog)

##Entre tornillo, tuerca, arandela y clavo
#print("Entre tornillo, tuerca, arandela y clavo")
#tornillo_fhog = m_hog(tornillo_gray[0])
#print(tornillo_fhog)
#clavo_fhog = m_hog(clavo_gray[0])
#print(clavo_fhog)
#tuerca_fhog = m_hog(tuerca_gray[0])
#print(tuerca_fhog)
#arandela_fhog = m_hog(arandela_gray[0])
#print(arandela_fhog)

#Haralick Textura
import mahotas

def haralick(image):
    feature = mahotas.features.haralick(image).mean(axis=0)
    return feature

##Entre 5 arandelas distintos
print("Entre 5 arandelas distintos")
for j in range(5):
    print(haralick(arandela_gray[j]))

##Entre 5 clavos distintos
print("Entre 5 clavos distintos")
for j in range(5):
    print(haralick(clavo_gray[j]))
          
##Entre 5 tornillos distintos
print("Entre 5 tornillos distintos")
for j in range(5):
    print(haralick(tornillo_gray[j]))

##Entre 5 tuercas distintos
print("Entre 5 tuercas distintos")
for j in range(5):
    print(haralick(tuerca_gray[j]))
          
##Entre tornillo, tuerca, arandela y clavo
print("Entre tornillo, tuerca, arandela y clavo")
print(haralick(tornillo_gray[0]))
print(haralick(clavo_gray[0]))
print(haralick(tuerca_gray[0]))
print(haralick(arandela_gray[0]))

##Momentos de Hu
#def hu_moments(image):
#    feature = cv2.HuMoments(cv2.moments(image)).flatten()
#    return feature

##Entre 5 arandelas distintos
#print("Entre 5 arandelas distintos")
#print(hu_moments(arandela_edge[0]))
#print(hu_moments(arandela_edge[1]))
#print(hu_moments(arandela_edge[2]))
#print(hu_moments(arandela_edge[3]))
#print(hu_moments(arandela_edge[4]))

##Entre 5 clavos distintos
#print("Entre 5 clavos distintos")
#print(hu_moments(clavo_edge[0]))
#print(hu_moments(clavo_edge[1]))
#print(hu_moments(clavo_edge[2]))
#print(hu_moments(clavo_edge[3]))
#print(hu_moments(clavo_edge[4]))

##Entre 5 tornillos distintos
#print("Entre 5 tornillos distintos")
#print(hu_moments(tornillo_edge[0]))
#print(hu_moments(tornillo_edge[1]))
#print(hu_moments(tornillo_edge[2]))
#print(hu_moments(tornillo_edge[3]))
#print(hu_moments(tornillo_edge[4]))

##Entre 5 tuercas distintos
#print("Entre 5 tuercas distintos")
#print(hu_moments(tuerca_edge[0]))
#print(hu_moments(tuerca_edge[1]))
#print(hu_moments(tuerca_edge[2]))
#print(hu_moments(tuerca_edge[3]))
#print(hu_moments(tuerca_edge[4]))

##Entre tornillo, tuerca, arandela y clavo
#print("Entre tornillo, tuerca, arandela y clavo")
#print(hu_moments(tornillo_edge[0]))
#print(hu_moments(tuerca_edge[0]))
#print(hu_moments(arandela_edge[0]))
#print(hu_moments(clavo_edge[0]))


