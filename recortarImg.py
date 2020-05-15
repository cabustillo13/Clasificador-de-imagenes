#NOTA: Consideramos que este programa solo es capaz de analizar un objeto dentro de la imagen
import cv2

imagen = cv2.imread("Data Base/YTrain/ZArandelas/photo0.jpg")
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(grises, 100, 200)

#Para OpenCV4
ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#Determinar coordenadas del rectangulo que encierra al objeto
x,y,w,h = cv2.boundingRect(ctns[0]) #x,y: coordenada de la parte izquierda de arriba. w: ancho y h:alto

crop_img = imagen[y:y+h, x:x+w]

path="/home/carlos/Documentos/Image-Classifier/"
fileb="prueba_imagenCortada.jpg"

cv2.imwrite(path + fileb, crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



