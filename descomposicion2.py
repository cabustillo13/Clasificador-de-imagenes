from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import io
#Las fotos de entrada estan en formato png o jpeg

prueba = './Data Base/YTrain/ZArandelas/photo0.jpg'
#prueba = './ejemplos/tornillo_prueba.jpg'

tornillo = io.imread(prueba)
#El modulo io tiene utilidades para leer y escribir imagenes en varios formatos.
#io.imread lectura y escritura de las imagenes via imread

#Redimensionamiento de la imagen de entrada
fixed_size = tuple((500, 400))
tornillo = cv2.resize(tornillo, fixed_size)

#Descomposicion de la imagen en matrices correspondientes a las 3 capas RGB previamente mencionadas.

R_tornillo = tornillo[:, :, 0]
G_tornillo = tornillo[:, :, 1]
B_tornillo = tornillo[:, :, 2]

f, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax in axes:
    ax.axis('off')
    
(ax_color, ax_B, ax_G, ax_R ) = axes

ax_color.imshow(np.stack([R_tornillo, G_tornillo, B_tornillo], axis=2))
ax_color.set_title('Las 3 capas Superpuestas')

ax_B.imshow(B_tornillo)
ax_B.set_title('Capa Azul')

ax_R.imshow(R_tornillo)
ax_R.set_title('Capa Roja')

ax_G.imshow(G_tornillo)
ax_G.set_title('Capa Verde')

plt.show()
