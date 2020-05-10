##PARA CORTAR VARIAS IMAGENES

import cv2
import numpy as np

##Carpeta donde quiero hacer el recorte de las imagenes
carpeta="Clavos"
#carpeta="Tuercas"
#carpeta="Arandelas"
#carpeta = "Tornillos"


##INICIALIZACION
path="/home/carlos/Documentos/Image-Classifier/Data Base/"
#path="/home/carlos/Documentos/Data Base/"
for i in range(100): #Arranca en 0 y termina en 100-1
    nombre= path+carpeta+"/"+"photo"+str(i)+".jpg"

    imagen = cv2.imread(nombre)

    #Determinar dimensiones de la imagen
    height, width, channels = imagen.shape
    
    #x marca hasta donde cortar (osea corta del pixel 0 hasta 700) 
    x=700
    crop_img = imagen[0:height, x:width]

    #Path de la nueva carpeta donde se guardaran las imagenes recortadas
    new_path= path+"Z"+carpeta+"/"+"photo"+str(i)+".jpg"

    cv2.imwrite(new_path, crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
