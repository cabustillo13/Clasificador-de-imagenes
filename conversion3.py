from skimage import color, img_as_float
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

#Conversion a escalas de grises haciendo uso del modulo de scikit
tornillo_float = img_as_float(tornillo)
sk_gray = color.rgb2gray(tornillo_float)

#Tambien se puede modificar manualmente los pesos de cada color en vez de utilizar los propuestos por la libreria
#Aca se colocan el mismo peso para los tres [1/3 1/3 1/3]
f, (ax1, ax0) = plt.subplots(1, 2, figsize=(16, 5))

ax1.imshow(tornillo)
ax1.set_title('Original')
ax0.imshow(sk_gray)
ax0.set_title('WG, WB y WR por skimage')
#Nota: WG, WB y WR son los coeficientes que definen la transformacion
plt.show()


#Bibliografia consultada
#https://pythoneyes.wordpress.com/2017/05/22/conversion-de-imagenes-rgb-a-escala-de-grises/
