#Esto es para el caso que no querer utilizar los pesos WG, WR y WB propuestos por el modulo -> sino para ingresarlos manualmente

from PIL import Image

foto = Image.open('./Data Base/YTrain/ZArandelas/photo0.jpg')
datos = foto.getdata()
#Para el calculo del promedio se utilizara la division entera con el operador de division doble "//" para evitar decimales
 
promedio = [(datos[x][0] + datos[x][1] + datos[x][2]) // 3 for x in range(len(datos))]
 
imagen_gris = Image.new('L', foto.size)
 
imagen_gris.putdata(promedio)
 
#Lo guarda con ese nombre
imagen_gris.save('./Figure_4.png')

foto.close()
 
imagen_gris.close() 

#Bibliografia consultada
#https://pythoneyes.wordpress.com/2017/05/22/conversion-de-imagenes-rgb-a-escala-de-grises/
