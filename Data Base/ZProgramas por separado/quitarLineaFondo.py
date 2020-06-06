##PARA CORTAR UNA SOLA IMAGEN

import cv2
import numpy as np

imagen = cv2.imread("photo.jpg")

#Determinar dimensiones de la imagen
height, width, channels = imagen.shape

#x marca hasta donde cortar (osea corta del pixel 0 hasta 700) 
x=700
crop_img = imagen[0:height, x:width]

path="/home/carlos/Documentos/Image-Classifier/Data Base/"
#path="/home/carlos/Documentos/Data Base/"
fileb="imagenCortada.jpg"

cv2.imwrite(path + fileb, crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


