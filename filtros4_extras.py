from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import filters 
from scipy.ndimage import convolve
#GAUSS
bright_square = np.zeros((7, 7), dtype=float)
bright_square[2:5, 2:5] = 1

sigma = 1
smooth = filters.gaussian(bright_square, sigma)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
ax0.imshow(bright_square)
ax0.set_title('Original')
ax1.imshow(smooth)
ax1.set_title('Filtro Gauss')
plt.show()

#DIFERENCIAL
vertical_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

horizontal_kernel = vertical_kernel.T

horizontal_gradient = convolve(bright_square.astype(float), horizontal_kernel)
vertical_gradient = convolve(bright_square.astype(float), vertical_kernel)

gradient = np.sqrt(horizontal_gradient**2 + vertical_gradient**2)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 5))
ax0.imshow(bright_square)
ax0.set_title('Original')
ax1.imshow(horizontal_gradient)
ax1.set_title('Kernel horizontal')
ax2.imshow(vertical_gradient)
ax2.set_title('Kernel vertical')
ax3.imshow(gradient)
ax3.set_title('Filtro Diferencial')
plt.show()
