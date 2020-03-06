from skimage import filters
from skimage.morphology import disk

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

#Gauss
bs0 = filters.gaussian(tornillo, sigma=1)
bs1 = filters.gaussian(tornillo, sigma=3)
bs2 = filters.gaussian(tornillo, sigma=5)
bs3 = filters.gaussian(tornillo, sigma=15)

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 5))
ax0.imshow(bs0)
ax0.set_title('$\sigma=1$')
ax1.imshow(bs1)
ax1.set_title('$\sigma=3$')
ax2.imshow(bs2)
ax2.set_title('$\sigma=5$')
ax3.imshow(bs2)
ax3.set_title('$\sigma=15$')
plt.show()
