import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['image.cmap'] = 'gray'

from mpl_toolkits.mplot3d import Axes3D

from skimage import io, color, img_as_float, filters
from skimage.feature import hog

import cv2
import mahotas

#Pre procesamiento
def img2grey(image, mode='sk'):
    if (mode=='sk'):
        gray = color.rgb2gray(image)
    elif (mode=='cv'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def normSize(image, size=(tuple((500, 400)))): #Paso del tamano 1080x1220 a 500x400
    image = cv2.resize(image, size)
    return image

#Filtracion
def imgClean(image, sigma=1, mode='sk'):
    if (mode == 'sk'):
        clean = filters.gaussian(image, sigma)
    elif (mode == 'cv'):
        clean = cv2.GaussianBlur(image, (3, 3), 0)
    return clean

def imgEdge(image, mode='sk'):
    if (mode == 'sk'):
        edge = filters.sobel(image)
    elif (mode == 'cv'):
        edge = cv2.Laplacian(image, cv2.CV_64F)
    return edge

#Segmentacion
def threshold(image, mode='sk'):
    if (mode == 'sk'):
        th = filters.threshold_isodata(image)
    elif (mode == 'cv'):
        ret, th = cv2.threshold(image, 0, 255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (image < th)

#Extraccion de Rasgos
def m_hog(image):
    feature = hog(image, block_norm='L2-Hys').ravel()
    return feature

def hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def haralick(image):
    feature= mahotas.features.haralick(image).mean(axis=0)
    return feature

def color_histogram(image, mask=None, bins=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    feature = hist.flatten()
    return feature

#Redimensionamiento
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

#Base de Datos
#tornillo = io.ImageCollection('./data/tornillos/*.png:./data/tornillos/*.jpg')
#tuerca = io.ImageCollection('./data/tuercas/*.png:./data/tuercas/*.jpg')
#arandela = io.ImageCollection('./data/arandelas/*.png:./data/arandelas/*.jpg')
#clavo = io.ImageCollection('./data/clavos/*.png:./data/clavos/*.jpg')

tornillo = io.ImageCollection('./Data Base/YTrain/ZTornillos/*.png:./Data Base/YTrain/ZTornillos/*.jpg')
tuerca = io.ImageCollection('./Data Base/YTrain/ZTuercas/*.png:./Data Base/YTrain/ZTuercas/*.jpg')
arandela = io.ImageCollection('./Data Base/YTrain/ZArandelas/*.png:./Data Base/YTrain/ZArandelas/*.jpg')
clavo = io.ImageCollection('./Data Base/YTrain/ZClavos/*.png:./Data Base/YTrain/ZClavos/*.jpg')

class Elemento:
    def __init__(self):
        self.label = None
        self.image = None
        self.feature = []
        self.distance = 0
        
def ft_extract(image):
    image = normSize(image)
    aux = img2grey(image, mode='cv')
    aux = imgClean(aux, mode='cv')
    aux = imgEdge(aux)
    # aux = threshold(aux, mode='cv')
    
    # image_fht = haralick(aux)
    image_fhm = hu_moments(aux)
    # image_fch = color_histogram(image)
    # image_fhog = m_hog(aux)

    # feature = np.hstack([image_fht, image_fhm, image_fhog])
    # med, dstd = stats(image_fhm)
    # feature = feature.reshape(1, -1)

    # return aux, [med, dstd]
    return aux, [image_fhm[0], image_fhm[1], image_fhm[3]]

#Analisis de datos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = []
i = 0

# Analisis de tornillos
iter = 0
for objeto in tornillo:
    data.append(Elemento())
    data[i].label = 'Tornillo'
    data[i].image, data[i].feature = ft_extract(objeto)
    # print(data[i].feature.shape)
    ax.scatter(data[i].feature[0], data[i].feature[1], data[i].feature[2], c='y', marker='o')
    i += 1
    iter += 1
print("Tornillos OK")

# Analisis de tuercas
iter = 0
for objeto in tuerca:
    data.append(Elemento())
    data[i].label = 'Tuerca'
    data[i].image, data[i].feature = ft_extract(objeto)
    ax.scatter(data[i].feature[0], data[i].feature[1], data[i].feature[2], c='r', marker='o')
    i += 1
    iter += 1
print("Tuercas OK")

# Analisis de arandelas
iter = 0
for objeto in arandela:
    data.append(Elemento())
    data[i].label = 'Arandela'
    data[i].image, data[i].feature = ft_extract(objeto)
    ax.scatter(data[i].feature[0], data[i].feature[1], data[i].feature[2], c='b', marker='o')
    i += 1
    iter += 1
print("Arandelas OK")

# Analisis de clavos
iter = 0
for objeto in clavo:
    data.append(Elemento())
    data[i].label = 'Clavo'
    data[i].image, data[i].feature = ft_extract(objeto)
    ax.scatter(data[i].feature[0], data[i].feature[1], data[i].feature[2], c='g', marker='o')
    i += 1
    iter += 1
print("Clavos OK")

ax.grid(True)
ax.set_title("Analisis de la base de datos")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

ax.set_xlabel('comp1')
ax.set_ylabel('comp2')
ax.set_zlabel('comp3')

##Si se quiere ver el "Elemento  evaluar" en el grafico su linea de codigo debe ir aca

plt.show()

print("Analisis completo")
#Cantidad de data
print("Cantidad de imagenes analizadas:")
print(len(data))

# Elemento a evaluar
test = Elemento()

#image = io.imread('./Data Base/YTest/ZClavos/photo99.jpg')
image = io.imread('./Data Base/YTrain/ZArandelas/photo0.jpg')


#image = io.imread('./ejemplos/arandela10_test.jpg') ##Esto es algo que podriamos variar por teclado
test.image, test.feature = ft_extract(image)
test.label = 'Arandela' # label inicial ##Esto es algo que podriamos variar por teclado

ax.scatter(test.feature[0], test.feature[1], test.feature[2], c='k', marker='o')
fig

#KNN
i = 0
sum = 0
for ft in data[0].feature:
        sum = sum + np.power(np.abs(test.feature[i] - ft), 2)
        i += 1
d = np.sqrt(sum)

for element in data:
    sum = 0
    i = 0
    for ft in (element.feature):
        sum = sum + np.power(np.abs((test.feature[i]) - ft), 2)
        i += 1
    
    element.distance = np.sqrt(sum)
    
    if (sum < d):
        d = sum
        test.label = element.label

print("Prediccion para K=1: ")    
print(test.label)

##Bubblesort
swap = True
while (swap):
    swap = False
    for i in range(1, len(data)-1) :
        if (data[i-1].distance > data[i].distance):
            aux = data[i]
            data[i] = data[i-1]
            data[i-1] = aux
            swap = True
print("Predicciones para K=10: ")            
k = 10
for i in range(k):
    print(data[i].label)


#K MEANS

##Entrenamiento

import random

tornillo_data = []
tuerca_data = []
arandela_data = []
clavo_data = []

for element in data:
    if (element.label == 'Tornillo'):
        tornillo_data.append(element)
    if (element.label == 'Tuerca'):
        tuerca_data.append(element)
    if (element.label == 'Arandela'):
        arandela_data.append(element)
    if (element.label == 'Clavo'):
        clavo_data.append(element)

tornillo_mean = list(random.choice(tornillo_data).feature)
tuerca_mean = list(random.choice(tuerca_data).feature)
arandela_mean = list(random.choice(arandela_data).feature)
clavo_mean = list(random.choice(clavo_data).feature)


fig_means = plt.figure()
ax = fig_means.add_subplot(111, projection='3d')

# fig_means, ax = plt.subplots()
ax.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='y', marker='o')
# ax.plot(b_mean[0], b_mean[1], 'o', color='yellow')
ax.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='r', marker='o')
# ax.plot(o_mean[0], o_mean[1], 'o', color='red')
ax.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='b', marker='o')
# ax.plot(l_mean[0], l_mean[1], 'o', color='blue')
ax.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='g', marker='o')
# ax.plot(l_mean[0], l_mean[1], 'o', color='blue')

ax.grid(True)
ax.set_title("Means")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

ax.set_xlabel('comp0')
ax.set_ylabel('comp1')
ax.set_zlabel('comp3')

plt.show()

##Asignacion, actualizacion y convergencia
tornillo_flag = True
tuerca_flag = True
arandela_flag = True
clavo_flag = True

tornillo_len = [0, 0, 0]
tuerca_len = [0, 0, 0]
arandela_len = [0, 0, 0]
clavo_len = [0, 0, 0]

iter = 0
#while (b_flag or o_flag or l_flag):
while (iter < 10):

    tornillo_data = []
    tuerca_data = []
    arandela_data = []
    clavo_data = []

    # ASIGNACION

    for element in data:
        sum_tornillo = 0
        sum_tuerca = 0
        sum_arandela = 0
        sum_clavo = 0

        for i in range(0, len(element.feature)-1):
            sum_tornillo += np.power(np.abs(tornillo_mean[i] - element.feature[i]), 2)
            sum_tuerca += np.power(np.abs(tuerca_mean[i] - element.feature[i]), 2)
            sum_arandela += np.power(np.abs(arandela_mean[i] - element.feature[i]), 2)
            sum_clavo += np.power(np.abs(clavo_mean[i] - element.feature[i]), 2)

        dist_tornillo = np.sqrt(sum_tornillo)
        dist_tuerca = np.sqrt(sum_tuerca)
        dist_arandela = np.sqrt(sum_arandela)
        dist_clavo = np.sqrt(sum_clavo)
        
        aux = dist_tornillo
        if (dist_tuerca < aux):
            aux = dist_tuerca
        if (dist_arandela < aux):
            aux = dist_arandela
        if (dist_clavo < aux):
            aux = dist_clavo
            
        if (aux == dist_tornillo):
            tornillo_data.append(element.feature)
        elif (aux == dist_tuerca):
            tuerca_data.append(element.feature)
        elif(aux == dist_arandela):
            arandela_data.append(element.feature)
        elif(aux == dist_clavo):
            clavo_data.append(element.feature)
            
    # ACTUALIZACION
    sum_tornillo = [0, 0, 0]
    for b in tornillo_data:
        sum_tornillo[0] += b[0]
        sum_tornillo[1] += b[1]
        sum_tornillo[2] += b[2]

    sum_tuerca = [0, 0, 0]
    for o in tuerca_data:
        sum_tuerca[0] += o[0]
        sum_tuerca[1] += o[1]
        sum_tuerca[2] += o[2]

    sum_arandela = [0, 0, 0]
    for l in arandela_data:
        sum_arandela[0] += l[0]
        sum_arandela[1] += l[1]
        sum_arandela[2] += l[2]

    sum_clavo = [0, 0, 0]
    for p in clavo_data:
        sum_clavo[0] += p[0]
        sum_clavo[1] += p[1]
        sum_clavo[2] += p[2]
        
    tornillo_mean[0] = sum_tornillo[0] / len(tornillo_data)
    tornillo_mean[1] = sum_tornillo[1] / len(tornillo_data)
    tornillo_mean[2] = sum_tornillo[2] / len(tornillo_data)

    tuerca_mean[0] = sum_tuerca[0] / len(tuerca_data)
    tuerca_mean[1] = sum_tuerca[1] / len(tuerca_data)
    tuerca_mean[2] = sum_tuerca[2] / len(tuerca_data)

    arandela_mean[0] = sum_arandela[0] / len(arandela_data)
    arandela_mean[1] = sum_arandela[1] / len(arandela_data)
    arandela_mean[2] = sum_arandela[1] / len(arandela_data)
    
    clavo_mean[0] = sum_clavo[0] / len(clavo_data)
    clavo_mean[1] = sum_clavo[1] / len(clavo_data)
    clavo_mean[2] = sum_clavo[1] / len(clavo_data)
    
    print("Tornillo  Tuerca  Arandela  Clavo")
    print(len(tornillo_data), len(tuerca_data), len(arandela_data), len(clavo_data))
    
    # CONVERGENCIA Y CONDICION DE SALIDA
    #print(len(banana_data), len(orange_data), len(lemon_data))
    """
    if (b_mean == b_len):
        b_flag = False
    else:
        b_len = b_mean
        
    if (o_mean == o_len):
        o_flag = False
    else:
        o_len = o_mean

    if (l_mean == l_len):
        l_flag = False
    else:
        l_len = l_mean
    """
    iter += 1
    
##Ubicacion de los means finales
ax.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='k', marker='o')
ax.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='k', marker='o')
ax.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='k', marker='o')
ax.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='k', marker='o')

print(len(tornillo_data), len(tuerca_data), len(arandela_data), len(clavo_data))
fig_means

##Mean mas cercano
sum_tornillo = 0
sum_tuerca = 0
sum_arandela = 0
sum_clavo = 0

for i in range(0, len(test.feature)-1):
    sum_tornillo += np.power(np.abs(test.feature[i] - tornillo_mean[i]), 2)
    sum_tuerca += np.power(np.abs(test.feature[i] - tuerca_mean[i]), 2)
    sum_arandela += np.power(np.abs(test.feature[i] - arandela_mean[i]), 2)
    sum_clavo += np.power(np.abs(test.feature[i] - clavo_mean[i]), 2)

dist_tornillo = np.sqrt(sum_tornillo)
dist_tuerca = np.sqrt(sum_tuerca)
dist_arandela = np.sqrt(sum_arandela)
dist_clavo = np.sqrt(sum_clavo)
print(dist_tornillo, dist_tuerca, dist_arandela, dist_clavo)

aux = dist_tornillo
if (dist_tuerca < aux):
    aux = dist_tuerca
if (dist_arandela < aux):
    aux = dist_arandela
if (dist_clavo < aux):
    aux = dist_clavo

if (aux == dist_tornillo):
    test.label = 'Tornillo'
elif (aux == dist_tuerca):
    test.label = 'Tuerca'
elif(aux == dist_arandela):
    test.label = 'Arandela'
elif(aux == dist_clavo):
    test.label = 'Clavo'

print("Prediccion: ")
print(test.label)
