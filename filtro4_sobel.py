import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import cv2

from skimage import io, color, img_as_float, filters

bright_square = np.zeros((7, 7), dtype=float)
bright_square[2:5, 2:5] = 1

kernel_sobel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

square_sobel = filters.sobel(bright_square)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
ax0.imshow(bright_square)
ax0.set_title('Original')
ax1.imshow(square_sobel)
ax1.set_title('Filtro Sobel')
plt.show()
