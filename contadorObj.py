#NOTA: Debido al modulo opencv, al correr cv2 algunas funciones (no todas) entran en conflicto con Kubuntu(la distro que yo uso). 
#entonces al correrlo quitar los doble # (##) y evaluelo con otra distro distinta a Ubuntu.

#NOTA:Consideramos que cada contorno encontrado equivale a un objeto dentro de la imagen

import cv2

#Aca cargamos la imagen
#imagen = cv2.imread('ejemplos/arandela_internet.jpg')
imagen = cv2.imread('ejemplos/arandelas2_internet.png')
#imagen = cv2.imread('ejemplos/tornillo_prueba.jpg')
#imagen = cv2.imread('ejemplos/cartas.png') #Por si quiere evaluar otra imagen
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#bordes = cv2.Canny(grises, 100, 200) #Para arandela_internet
#bordes = cv2.Canny(grises, 10, 800) #Para tornillo_prueba
bordes = cv2.Canny(grises, 10, 1300) #Para arandelas2_internet

#Para OpenCV4
ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imagen, ctns, -1, (0,0,255), 2)
#Muestra por consola
print('Numero de contornos encontrados: ', len(ctns))
#Incorpora un texto en la parte superior izquierda a la propia imagen
texto = 'Contornos encontrados: '+ str(len(ctns))

cv2.putText(imagen, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(255, 0, 0), 1)

path="/home/carlos/Documentos/Image-Classifier/ejemplos/"
fileb="imagenContorno_prueba.jpg"

cv2.imwrite(path + fileb, imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
