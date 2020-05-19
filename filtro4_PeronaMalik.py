import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

#Inicializacion
image_file = './Data Base/YTrain/ZArandelas/photo0.jpg'
#image_file = './ejemplos/tornillo_prueba.jpg'

iterations = 30
delta = 0.14
kappa = 15

#Convertir la imagen de entrada
im = misc.imread(image_file, flatten=True)
im = im.astype('float64')

#Condicion inicial
u = im

# Distancia al pixel central
dx = 1
dy = 1
dd = np.sqrt(2)

#2D diferentes finitas ventanas
windows = [
    np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
    ),
    np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
]

for r in range(iterations):
    #Aproximacion de gradientes
    nabla = [ ndimage.filters.convolve(u, w) for w in windows ]

    #Aproximacion de la funcion de difusion
    diff = [ 1./(1 + (n/kappa)**2) for n in nabla]

    #Actualizar imagen
    terms = [diff[i]*nabla[i] for i in range(4)]
    terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
    u = u + delta*(sum(terms))


# Kernel para el gradiente en la direccion x
Kx = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
)
# Kernel para el gradiente en la direccion y
Ky = np.array(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
)
#Aplicar kernel a la imagen
Ix = ndimage.filters.convolve(u, Kx)
Iy = ndimage.filters.convolve(u, Ky)

#Retorna (Ix, Iy)
G = np.hypot(Ix, Iy)

plt.subplot(1, 3, 1), plt.imshow(im, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('1')
#La 1 es la imagen original

plt.subplot(1, 3, 2), plt.imshow(u, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('2')
#La 2 es despues de la difusion

plt.subplot(1, 3, 3), plt.imshow(G, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('3')
#Es el gradiente despues de la difusion
plt.show() 

#Bibliografia consultada
#https://github.com/fubel/PeronaMalikDiffusion/blob/master/main.py
