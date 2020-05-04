from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import io
#Las fotos de entrada estan en formato png o jpeg

prueba = './ejemplos/tornillo_prueba.jpg'
tornillo = io.imread(prueba)
#print(tornillo.shape)
#El modulo io tiene utilidades para leer y escribir imagenes en varios formatos.
#io.imread lectura y escritura de las imagenes via imread

#Redimensionamiento de la imagen de entrada
fixed_size = tuple((400, 300))
tornillo = cv2.resize(tornillo, fixed_size)

print(tornillo.shape)
#Devuelve la estructura de la imagen.En este caso nos devolvera (300,400,3).
#Es importante aclarar que mas adelante veremos que la imagen la podemos describir como un array de de esas 3 capas (RGB).
plt.imshow(tornillo)
plt.show()
#Muestra por pantalla la nueva imagen redimensionada

#Bibliografia consultada
#https://scikit-image.org/docs/stable/auto_examples/data/plot_specific.html#sphx-glr-auto-examples-data-plot-specific-py
