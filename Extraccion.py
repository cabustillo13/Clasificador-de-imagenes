#####################################################################################################################################
##Representacion de Histograma de color para una imagen filtrada
#####################################################################################################################################
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from skimage import io, color, img_as_float, img_as_ubyte, filters
import cv2

def escala_grises(image):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gris

def normalizacion(image):
    image = cv2.resize(image, (500, 400))
    return image

def gaussian(image);
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def sobel(image):
    image = filters.sobel(image)
    return image

img = io.imread('./Data Base/YTrain/ZArandelas/photo0.jpg')
tornillo = escala_grises(img)
tornillo = normalizacion(tornillo)

bg = gaussian(tornillo)
bc = sobel(bg)

fig, (ax2, ax1, ax0) = plt.subplots(1, 3, figsize=(16, 5))
ax0.imshow(bc)
ax0.hlines(bc.shape[0]//2, 0, bc.shape[1], color='C3')
ax0.set_title('Filtro Gauss + Sobel')
ax1.plot(bc[bc.shape[0]//2, :], color='C3')
ax1.set_title('Linea divisoria central')

ax2.hist(bc.ravel(), bins=32, range=[0, 256], color='C3')
ax2.set_xlim(0, 256);
ax2.set_title('Histograma de color para imagen filtrada')
plt.show()

#####################################################################################################################################
##Comparacion de caracteristicas Hu, HOG y Haralick para las distintas piezas de ferreteria
#####################################################################################################################################
#Determinacion de la base de datos
tornillo = io.ImageCollection('./Hog, Haralick y Hu/No Cortadas/Tornillos/*.png:./Hog, Haralick y Hu/No Cortadas/Tornillos/*.jpg')
tuerca = io.ImageCollection('./Hog, Haralick y Hu/No Cortadas/Tuercas/*.png:./Hog, Haralick y Hu/No Cortadas/Tuercas/*.jpg')
arandela = io.ImageCollection('./Hog, Haralick y Hu/No Cortadas/Arandelas/*.png:./Hog, Haralick y Hu/No Cortadas/Arandelas/*.jpg')
clavo = io.ImageCollection('./Hog, Haralick y Hu/No Cortadas/Clavos/*.png:./Hog, Haralick y Hu/No Cortadas/Clavos/*.jpg')

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
    aux = normalizacion(tornillo[i])
    #aux = gaussian(aux)
    tornillo_n.append(aux)
    tornillo_gray.append(escala_grises(tornillo_n[i]))
    tornillo_edge.append(sobel(tornillo_gray[i]))

i = 0
for i in range(0, len(tuerca)):
    aux = normalizacion(tuerca[i])
    #aux = gaussian(aux)
    tuerca_n.append(aux)
    tuerca_gray.append(escala_grises(tuerca_n[i]))
    tuerca_edge.append(sobel(tuerca_gray[i]))

i = 0
for i in range(0, len(arandela)):
    aux = normalizacion(arandela[i])
    #aux = gaussian(aux)
    arandela_n.append(aux)
    arandela_gray.append(escala_grises(arandela_n[i]))
    arandela_edge.append(sobel(arandela_gray[i]))
    
i = 0
for i in range(0, len(clavo)):
    aux = normalizacion(clavo[i])
    #aux = gaussian(aux)
    clavo_n.append(aux)
    clavo_gray.append(escala_grises(clavo_n[i]))
    clavo_edge.append(sobel(clavo_gray[i]))
    
#HOG: Histograma de gradientes orientados
#from skimage.feature import hog

#def hog(image):
#    caracteristica = hog(image, block_norm='L2-Hys').ravel()
#    return caracteristica

##Entre 5 arandelas distintos
#print("Entre 5 arandelas distintos")
#for j in range(5):
#    print(hog(arandela_gray[j]))

##Entre 5 clavos distintos
#print("Entre 5 clavos distintos")
#for j in range(5):
#    print(hog(clavo_gray[j]))
          
##Entre 5 tornillos distintos
#print("Entre 5 tornillos distintos")
#for j in range(5):
#    print(hog(tornillo_gray[j]))

##Entre 5 tuercas distintos
#print("Entre 5 tuercas distintos")
#for j in range(5):
#    print(hog(tuerca_gray[j]))
          
##Entre tornillo, tuerca, arandela y clavo
#print("Entre tornillo, clavo, tuerca y arandela")
#print(hog(tornillo_gray[0]))
#print(hog(clavo_gray[0]))
#print(hog(tuerca_gray[0]))
#print(hog(arandela_gray[0]))

#Haralick Textura
import mahotas

def haralick(image):
    caracteristica = mahotas.features.haralick(image).mean(axis=0)
    return caracteristica

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
#def hu(image):
#    caracteristica = cv2.HuMoments(cv2.moments(image)).flatten()
#    return caracteristica

##Entre 5 arandelas distintos
#print("Entre 5 arandelas distintos")
#print(hu(arandela_edge[0]))
#print(hu(arandela_edge[1]))
#print(hu(arandela_edge[2]))
#print(hu(arandela_edge[3]))
#print(hu(arandela_edge[4]))

##Entre 5 clavos distintos
#print("Entre 5 clavos distintos")
#print(hu(clavo_edge[0]))
#print(hu(clavo_edge[1]))
#print(hu(clavo_edge[2]))
#print(hu(clavo_edge[3]))
#print(hu(clavo_edge[4]))

##Entre 5 tornillos distintos
#print("Entre 5 tornillos distintos")
#print(hu(tornillo_edge[0]))
#print(hu(tornillo_edge[1]))
#print(hu(tornillo_edge[2]))
#print(hu(tornillo_edge[3]))
#print(hu(tornillo_edge[4]))

##Entre 5 tuercas distintos
#print("Entre 5 tuercas distintos")
#print(hu(tuerca_edge[0]))
#print(hu(tuerca_edge[1]))
#print(hu(tuerca_edge[2]))
#print(hu(tuerca_edge[3]))
#print(hu(tuerca_edge[4]))

##Entre tornillo, tuerca, arandela y clavo
#print("Entre tornillo, tuerca, arandela y clavo")
#print(hu(tornillo_edge[0]))
#print(hu(tuerca_edge[0]))
#print(hu(arandela_edge[0]))
#print(hu(clavo_edge[0]))

#####################################################################################################################################
##Reduccion de dimensionalidad para Hu, Haralick y HOG
#####################################################################################################################################
import matplotlib.patches as mpatches

#Estadistica -> Aca vamos a anlizar la frecuencia de aparicion para cada pieza, y para eso hacemos uso de la media aritmetica y la desviacion estandar
def estadistica(array):
    
    sum = 0
    for value in array:
        sum += value
    media = sum / len(array)
    sum = 0
    for value in array:
        sum += np.power((value - media), 2)
    desviacion = np.sqrt(sum / (len(array) - 1))
    
    return media, desviacion

#HOG 
grafico_hog, ax = plt.subplots()

for objeto in tornillo_gray:
    tornillo_fhog = hog(objeto)
    media, desviacion = estadistica(tornillo_fhog)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in tuerca_gray:
    tuerca_fhog = hog(objeto)
    media, desviacion = estadistica(tuerca_fhog)
    ax.plot(media, desviacion, 'o', color='red')
    
for objeto in arandela_gray:
    arandela_fhog = hog(objeto)
    media, desviacion = estadistica(arandela_fhog)
    ax.plot(media, desviacion, 'o', color='blue')

for objeto in clavo_gray:
    clavo_fhog = hog(objeto)
    media, desviacion = estadistica(clavo_fhog)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de dimensionalidad para HOG")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')

plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show()

#Hu
grafico_hu, ax = plt.subplots()

for objeto in tornillo_edge:
    tornillo_fhm = hu(objeto)
    media, desviacion = estadistica(tornillo_fhm)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in tuerca_edge:
    tuerca_fhm = hu(objeto)
    media, desviacion = estadistica(tuerca_fhm)
    ax.plot(media, desviacion, 'o', color='red')
    
for objeto in arandela_edge:
    arandela_fhm = hu(objeto)
    media, desviacion = estadistica(arandela_fhm)
    ax.plot(media, desviacion, 'o', color='blue')

for objeto in clavo_edge:
    clavo_fhm = hu(objeto)
    media, desviacion = estadistica(clavo_fhm)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de Dimensionalidad para Hu")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label = 'Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show()

#Haralick
grafica_haralick, ax = plt.subplots()

for objeto in tornillo_gray:
    tornillo_fht = haralick(objeto)
    media, desviacion = estadistica(tornillo_fht)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in tuerca_gray:
    tuerca_fht = haralick(objeto)
    media, desviacion = estadistica(tuerca_fht)
    ax.plot(media, desviacion, 'o', color='red')
    
for objeto in arandela_gray:
    arandela_fht = haralick(objeto)
    media, desviacion = estadistica(arandela_fht)
    ax.plot(media, desviacion, 'o', color='blue')

for objeto in clavo_gray:
    clavo_fht = haralick(objeto)
    media, desviacion = estadistica(clavo_fht)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de Dimensionalidad para Haralick")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label = 'Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show()

