import numpy as np
import cv2

##Seleccione que carpeta quiere: YTrain o YTest
carpeta="YTest"
#carpeta="YTrain"

##Seleccione la pieza
#pieza="Tornillos"
#pieza="Clavos"
pieza="Tuercas"
#pieza="Arandelas"

##Numero de elementos de la carpeta YTrain o YTest 
#numero= 150 #Para YTrain
numero=38  #Para YTest

path= "/home/carlos/Documentos/Image-Classifier/Data Base/"+carpeta+"/Z"+pieza+"/"

for i in range(numero+1):
    #Cargar la imagen
    foto = path+"photo"+str(i)+".jpg"
    imgo = cv2.imread(foto)
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
    background[np.where((background > [100,100,100]).all(axis = 2))] = [255,255,255] #[100,100,100] color de fondo de la caja donde tome las fotos

    #Agregar el fondo y la imagen
    final = background + img1

    #Tambien se puede suavizar los bordes de la imagen
    nuevo_nombre = "/home/carlos/Documentos/Image-Classifier/Data Base/"+carpeta+"/Y"+pieza+"/"+"photo"+str(i)+".jpg"
    cv2.imwrite(nuevo_nombre, final )
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
