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

img = io.imread('./Data Base/YTrain/ZArandelas/photo0.jpg')
#img = io.imread('./ejemplos/tornillo_prueba.jpg')

tornillo = img2gray(img)
tornillo = normSize(tornillo)

bg = imgClean(tornillo)
bc = imgEdge(bg)

fig, (ax2, ax1, ax0) = plt.subplots(1, 3, figsize=(16, 5))
ax0.imshow(bc)
ax0.hlines(bc.shape[0]//2, 0, bc.shape[1], color='C3')
ax0.set_title('Filtrado')
ax1.plot(bc[bc.shape[0]//2, :], color='C3')
ax1.set_title('Linea central')

ax2.hist(bc.ravel(), bins=32, range=[0, 256], color='C3')
ax2.set_xlim(0, 256);
ax2.set_title('Histograma')
plt.show()
