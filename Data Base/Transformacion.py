import cv2
import numpy as np

#####################################################################################################################################
##Quitar linea de fondo
#####################################################################################################################################
nombre = raw_input("Ingrese nombre de la imagen: ")
path="/home/carlos/Documentos/Image-Classifier/Data Base/Evaluacion/"+str(nombre)+".jpg"
imagen = cv2.imread(path)

#Determinar dimensiones de la imagen
height, width, channels = imagen.shape

#x marca hasta donde cortar (osea corta del pixel 0 hasta 700) 
x=700
crop_img = imagen[0:height, x:width]

new_path="/home/carlos/Documentos/Image-Classifier/Data Base/YEvaluacion/"+str(nombre)+".jpg"
cv2.imwrite(new_path, crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#####################################################################################################################################
##Redimension de la imagen
#####################################################################################################################################

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from skimage import io
#Las fotos de entrada estan en formato png o jpeg

tornillo = io.imread(new_path)
#print(tornillo.shape) -> Esto te devuelve el ancho, alto y los canales
#El modulo io tiene utilidades para leer y escribir imagenes en varios formatos.
#io.imread lectura y escritura de las imagenes via imread

#Redimensionamiento de la imagen de entrada
fixed_size = tuple((500, 400))
tornillo = cv2.resize(tornillo, fixed_size)

print(tornillo.shape)
#Devuelve la estructura de la imagen.En este caso nos devolvera (400,500,3).
#Es importante aclarar que mas adelante veremos que la imagen la podemos describir como un array de de esas 3 capas (RGB).
plt.imshow(tornillo)
plt.show()
#Muestra por pantalla la nueva imagen redimensionada

#Bibliografia consultada
#https://scikit-image.org/docs/stable/auto_examples/data/plot_specific.html#sphx-glr-auto-examples-data-plot-specific-py

#####################################################################################################################################
##Quitar fondo -> convertir el fondo a fondo totalmente blanco
#####################################################################################################################################
#Cargar la imagen
imgo = cv2.imread(new_path)
height, width = imgo.shape[:2]

#Crear una mascara
mask = np.zeros(imgo.shape[:2],np.uint8)

#Grabar el corte del objeto
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#Hard Coding the Rect The object must lie within this rect.
rect = (1079,1079,width-1,height-1)
cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = imgo*mask[:,:,np.newaxis]

#Obtener el fondo
background = imgo - img1

#Cambiar todos los pixeles en el fondo que no sean de negro a blanco
background[np.where((background > [100,100,100]).all(axis = 2))] = [255,255,255] #[100,100,100] color de fondo (de la mascara) de la caja donde tome las fotos

#Agregar el fondo y la imagen
final = background + img1

#Tambien se puede suavizar los bordes de la imagen
cv2.imwrite(new_path, final )

cv2.waitKey(0)
cv2.destroyAllWindows()
