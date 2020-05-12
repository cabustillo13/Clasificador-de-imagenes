#Proyecto Final de Inteligencia Artificial - Universidad Nacional de Cuyo
#Carlos Alberto Bustillo Lopez, Legajo 11586
#Ingenieria Mecatronica, Marzo de 2020

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

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

##MODIFICAR ESTO osea las dimensiones
def normSize(image, size=(tuple((500, 400)))):
#def normSize(image, size=(tuple((1080, 1220)))):
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

#Extraccion de rasgos
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

#Reduccion de dimensiones
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

#Metodos
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
    # image = imgClean(image, mode='cv')
    
    # image_fht = haralick(aux)
    image_fhm = hu_moments(aux)
    # image_fch = color_histogram(image)
    # image_fhog = m_hog(aux)

    # feature = np.hstack([image_fht, image_fhm, image_fhog])
    # med, dstd = stats(image_fch)
    # feature = feature.reshape(1, -1)

    # return aux, [med, dstd]
    return aux, [image_fhm[0], image_fhm[1], image_fhm[3]]

#Analisis de la base de datos
##Training base de datos
def data_analysis():

    #tornillo = io.ImageCollection('./data/tornillos/*.png:./data/tornillos/*.jpg')
    #tuerca = io.ImageCollection('./data/tuercas/*.png:./data/tuercas/*.jpg')
    #arandela = io.ImageCollection('./data/arandelas/*.png:./data/arandelas/*.jpg')
    #clavo = io.ImageCollection('./data/clavos/*.png:./data/clavos/*.jpg')

    tornillo = io.ImageCollection('./Data Base/YTrain/ZTornillos/*.png:./Data Base/YTrain/ZTornillos/*.jpg')
    tuerca = io.ImageCollection('./Data Base/YTrain/ZTuercas/*.png:./Data Base/YTrain/ZTuercas/*.jpg')
    arandela = io.ImageCollection('./Data Base/YTrain/ZArandelas/*.png:./Data Base/YTrain/ZArandelas/*.jpg')
    clavo = io.ImageCollection('./Data Base/YTrain/ZClavos/*.png:./Data Base/YTrain/ZClavos/*.jpg')
    
    data = []
    i = 0

    # Analisis de tornillos en base de datos
    iter = 0
    for objeto in tornillo:
        data.append(Elemento())
        data[i].label = 'tornillo'
        data[i].image, data[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Tornillo OK")

    # Analisis de tuercas en base de datos
    iter = 0
    for objeto in tuerca:
        data.append(Elemento())
        data[i].label = 'tuerca'
        data[i].image, data[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Tuerca OK")

    # Analisis de arandelas en la base de datos
    iter = 0
    for objeto in arandela:
        data.append(Elemento())
        data[i].label = 'arandela'
        data[i].image, data[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Arandela OK")
    
    # Analisis de clavos en la base de datos
    iter = 0
    for objeto in clavo:
        data.append(Elemento())
        data[i].label = 'clavo'
        data[i].image, data[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Clavo OK")

    print("Analisis de los objetos completo")
    return data

##Testing base de datos
def test_analysis():

    #tornillo_test = io.ImageCollection('./testeo/tornillos/*.png:./testeo/tornillos/*.jpg')
    #tuerca_test = io.ImageCollection('./testeo/tuercas/*.png:./testeo/tuercas/*.jpg')
    #arandela_test = io.ImageCollection('./testeo/arandelas/*.png:./testeo/arandelas/*.jpg')
    #clavo_test = io.ImageCollection('./testeo/clavos/*.png:./testeo/clavos/*.jpg')

    tornillo_test = io.ImageCollection('./Data Base/YTest/ZTornillos/*.png:./Data Base/YTest/ZTornillos/*.jpg')
    tuerca_test = io.ImageCollection('./Data Base/YTest/ZTuercas/*.png:./Data Base/YTest/ZTuercas/*.jpg')
    arandela_test = io.ImageCollection('./Data Base/YTest/ZArandelas/*.png:./Data Base/YTest/ZArandelas/*.jpg')
    clavo_test = io.ImageCollection('./Data Base/YTest/ZClavos/*.png:./Data Base/YTest/ZClavos/*.jpg')
    
    test = []
    i = 0

    # Analisis de tornillos en base de datos
    iter = 0
    for objeto in tornillo_test:
        test.append(Elemento())
        test[i].label = 'tornillo'
        test[i].image, test[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Tornillo OK")

    # Analisis de tuercas en base de datos
    iter = 0
    for objeto in tuerca_test:
        test.append(Elemento())
        test[i].label = 'tuerca'
        test[i].image, test[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Tuerca OK")

    # Analisis de arandelas en la base de datos
    iter = 0
    for objeto in arandela_test:
        test.append(Elemento())
        test[i].label = 'arandela'
        test[i].image, test[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Arandela OK")
    
    # Analisis de clavos en la base de datos
    iter = 0
    for objeto in clavo_test:
        test.append(Elemento())
        test[i].label = 'clavo'
        test[i].image, test[i].feature = ft_extract(objeto)
        i += 1
        iter += 1
    print("Clavo OK")

    print("Testeo de los objetos completo")
    return test

#KNN
# NOTA IMPORTANTE: Antes de ejecutar este bloque actualizar base de datos test
def knn(k, data, test):

    correct = 0

    for t in test:

        for element in data:
            sum = 0
            i = 0
            for ft in (element.feature):
                sum = sum + np.power(np.abs((t.feature[i]) - ft), 2)
                i += 1

            element.distance = np.sqrt(sum)

        # Bubblesort
        swap = True
        while (swap):
            swap = False
            for i in range(1, len(data)-1) :
                if (data[i-1].distance > data[i].distance):
                    aux = data[i]
                    data[i] = data[i-1]
                    data[i-1] = aux
                    swap = True

        eval = [0, 0, 0, 0]

        for i in range(0, k):

            if (data[i].label == 'tornillo'):
                eval[0] += 10

            if (data[i].label == 'tuerca'):
                eval[1] += 10

            if (data[i].label == 'arandela'):
                eval[2] += 10
                
            if (data[i].label == 'clavo'):
                eval[3] += 10
                

        aux = eval[0]
        if (aux < eval[1]):
            aux = eval[1]
        if (aux < eval[2]):
            aux = eval[2]
        if (aux < eval[3]):
            aux = eval[3]

        if (aux == eval[0]):
            label = 'tornillo'
        if (aux == eval[1]):
            label = 'tuerca'
        if (aux == eval[2]):
            label = 'arandela'
        if (aux == eval[3]):
            label = 'clavo'

        if (t.label == label):
            correct += 1
         
    return correct

##Rendimiento KNN - Maldicion de dimensionalidad
data = data_analysis()
test = test_analysis()

#MAX = 50
MAX = 150

ans = []

for k in range(1, MAX):
    ans.append(knn(k, data, test))

for i in range(0, len(ans)-1):
    ans[i] = ans[i] * 100 / len(test)

fig, ax = plt.subplots()
ax.plot(ans)
ax.grid(True)
ax.set_title('Rendimiento vrs K')
plt.ylabel('Predicciones correctas (%)')
plt.xlabel('K')
plt.show()

#K MEANS
##Training 
import random

def kmeans_train(data):

    tornillo_data = []
    tuerca_data = []
    arandela_data = []
    clavo_data = []

    # MEANS INICIALES
    for element in data:
        if (element.label == 'tornillo'):
            tornillo_data.append(element)
        if (element.label == 'tuerca'):
            tuerca_data.append(element)
        if (element.label == 'arandela'):
            arandela_data.append(element)
        if (element.label == 'clavo'):
            clavo_data.append(element)

    tornillo_mean = list(random.choice(tornillo_data).feature)
    tuerca_mean = list(random.choice(tuerca_data).feature)
    arandela_mean = list(random.choice(arandela_data).feature)
    clavo_mean = list(random.choice(clavo_data).feature)

    tornillo_flag = True
    tuerca_flag = True
    arandela_flag = True
    clavo_flag = True

    tornillo_len = [0, 0, 0]
    tuerca_len = [0, 0, 0]
    arandela_len = [0, 0, 0]
    clavo_len = [0, 0, 0]

    iter = 0

    # while (b_flag or o_flag or l_flag):
    while (iter < 20):

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
        for obj1 in tornillo_data:
            sum_tornillo[0] += obj1[0]
            sum_tornillo[1] += obj1[1]
            sum_tornillo[2] += obj1[2]

        sum_tuerca = [0, 0, 0]
        for obj2 in tuerca_data:
            sum_tuerca[0] += obj2[0]
            sum_tuerca[1] += obj2[1]
            sum_tuerca[2] += obj2[2]

        sum_arandela = [0, 0, 0]
        for obj3 in arandela_data:
            sum_arandela[0] += obj3[0]
            sum_arandela[1] += obj3[1]
            sum_arandela[2] += obj3[2]
            
        sum_clavo = [0, 0, 0]
        for obj4 in clavo_data:
            sum_clavo[0] += obj4[0]
            sum_clavo[1] += obj4[1]
            sum_clavo[2] += obj4[2]

        tornillo_mean[0] = sum_tornillo[0] / len(tornillo_data)
        tornillo_mean[1] = sum_tornillo[1] / len(tornillo_data)
        tornillo_mean[2] = sum_tornillo[2] / len(tornillo_data)

        tuerca_mean[0] = sum_tuerca[0] / len(tuerca_data)
        tuerca_mean[1] = sum_tuerca[1] / len(tuerca_data)
        tuerca_mean[2] = sum_tuerca[2] / len(tuerca_data)

        arandela_mean[0] = sum_arandela[0] / len(arandela_data)
        arandela_mean[1] = sum_arandela[1] / len(arandela_data)
        arandela_mean[2] = sum_arandela[2] / len(arandela_data) #Corregi esto
        
        clavo_mean[0] = sum_clavo[0] / len(clavo_data)
        clavo_mean[1] = sum_clavo[1] / len(clavo_data)
        clavo_mean[2] = sum_clavo[2] / len(clavo_data) #Corregi esto
        # print(len(banana_data), len(orange_data), len(lemon_data))

        # CONVERGENCIA Y CONDICION DE SALIDA
        # print(len(banana_data), len(orange_data), len(lemon_data))

        if (tornillo_mean == tornillo_len):
            tornillo_flag = False
        else:
            tornillo_len = tornillo_mean

        if (tuerca_mean == tuerca_len):
            tuerca_flag = False
        else:
            tuerca_len = tuerca_mean

        if (arandela_mean == arandela_len):
            arandela_flag = False
        else:
            arandela_len = arandela_mean
            
        if (clavo_mean == clavo_len):
            clavo_flag = False
        else:
            clavo_len = clavo_mean

        iter += 1
        
    return [tornillo_mean, tuerca_mean, arandela_mean, clavo_mean]

def kmeans(test, means):
    
    tornillo_mean = means[0]
    tuerca_mean = means[1]
    arandela_mean = means[2]
    clavo_mean = means[3]
    
    correct = 0

    for t in test:

        sum_tornillo = 0
        sum_tuerca = 0
        sum_arandela = 0
        sum_clavo = 0

        for i in range(0, len(t.feature)-1):
            sum_tornillo += np.power(np.abs(t.feature[i] - tornillo_mean[i]), 2)
            sum_tuerca += np.power(np.abs(t.feature[i] - tuerca_mean[i]), 2)
            sum_arandela += np.power(np.abs(t.feature[i] - arandela_mean[i]), 2)
            sum_clavo += np.power(np.abs(t.feature[i] - clavo_mean[i]), 2)

        dist_tornillo = np.sqrt(sum_tornillo)
        dist_tuerca = np.sqrt(sum_tuerca)
        dist_arandela = np.sqrt(sum_arandela)
        dist_clavo = np.sqrt(sum_clavo)
        # print(dist_tornillo, dist_tuerca, dist_arandela,dista_clavo)

        aux = dist_tornillo
        if (dist_tuerca < aux):
            aux = dist_tuerca
        if (dist_arandela < aux):
            aux = dist_arandela
        if (dist_clavo < aux):
            aux = dist_clavo

        if (aux == dist_tornillo):
            label = 'tornillo'
        if (aux == dist_tuerca):
            label = 'tuerca'
        if (aux == dist_arandela):
            label = 'arandela'
        if (aux == dist_clavo):
            label = 'clavo'
        
        if (t.label == label):
            correct += 1
    
    return correct

##Rendimiento
data = data_analysis()
test = test_analysis()

means = kmeans_train(data)

#MAX = 50
MAX = 150


ans = []

for i in range(0, MAX):
    ans.append(kmeans(test, means))
    
for i in range(0, len(ans)):
    ans[i] = ans[i] * 100 / len(test)

fig, ax = plt.subplots()
ax.plot(ans)
ax.grid(True)
ax.set_title('Rendimiento en diferentes ejecuciones')
plt.ylabel('Predicciones correctas (%)')
plt.xlabel('# de ejecucion')
plt.show()
