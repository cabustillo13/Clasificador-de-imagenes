from skimage import color, img_as_float
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import io
#Las fotos de entrada estan en formato png o jpeg

prueba = './ejemplos/tornillo_prueba.jpg'
tornillo = io.imread(prueba)
#El modulo io tiene utilidades para leer y escribir imagenes en varios formatos.
#io.imread lectura y escritura de las imagenes via imread

#Redimensionamiento de la imagen de entrada
fixed_size = tuple((400, 300))
tornillo = cv2.resize(tornillo, fixed_size)

#Conversion a escalas de grises haciendo uso del modulo de scikit
tornillo_float = img_as_float(tornillo)
sk_gray = color.rgb2gray(tornillo_float)

#Tambien se puede modificar manualmente los pesos de cada color
#Este ejemplo tiene una implementacion para python3
#gray = tornillo_float @ [0.299, 0.587, 0.114]
# Esos valores son el criterio utilizado en la TV para senales a color
#gray = tornillo_float @ [1/3, 1/3, 1/3]
#Aca se colocan el mismo peso para los tres

plt.imshow(sk_gray)
plt.show()


#Bibliografia consultada
#https://pythoneyes.wordpress.com/2017/05/22/conversion-de-imagenes-rgb-a-escala-de-grises/
