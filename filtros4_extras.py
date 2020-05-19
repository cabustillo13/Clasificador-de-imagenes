from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import filters, io, color

#GAUSS
prueba = './ejemplos/tornillo_prueba.jpg'
image = io.imread(prueba)
gris = color.rgb2gray(image)

laplace = filters.laplace(gris)
median = filters.median(gris)
frangi = filters.frangi(gris)
prewitt = filters.prewitt(gris)
 
f, (ax0, ax1,ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(16, 5))
ax0.imshow(image)
ax0.set_title('Original')

ax1.imshow(frangi)
ax1.set_title('Filtro Frangi')

ax2.imshow(prewitt)
ax2.set_title('Filtro Prewitt')

ax3.imshow(laplace)
ax3.set_title('Filtro Laplace')

ax4.imshow(median)
ax4.set_title('Filtro Median')
plt.show()

#Bibliografia consultada
#https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel
