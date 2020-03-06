from PIL import Image

foto = Image.open('./ejemplos/tornillo_prueba.jpg')
datos = foto.getdata()
#Para el calculo del promedio se utilizara la division entera con el operador de division doble "//" para evitar decimales
 
promedio = [(datos[x][0] + datos[x][1] + datos[x][2]) // 3 for x in range(len(datos))]
 
imagen_gris = Image.new('L', foto.size)
 
imagen_gris.putdata(promedio)
 
#Lo guarda con ese nombre
imagen_gris.save('./ejemplos/conversion3_extra.jpg')

foto.close()
 
imagen_gris.close() 
