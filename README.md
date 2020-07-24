# Clasificador-de-imagenes

## Clasificador de piezas de ferretería (tornillos, clavos, arandelas y tuercas) mediante visión artificial.

**Proyecto de Inteligencia Artificial 1 - Universidad Nacional de Cuyo.**

Este proyecto de visión artificial consiste en la clasificación de piezas metálicas (tornillos, clavos, tuercas y arandelas). 
El programa consta de una etapa de: Adquisición de imágenes (a través de una caja cerrada con fondo blanco, con iluminación frontal y comunicación por IP 
entre cámara y computadora); Transformación (para tener una imagen normalizada); Adaptación y preprocesamiento; Filtración (se utilizo filtro Gauss y Sobel) 
y segmentación; Extracción de rasgos (se utilizo Momentos de Hu) y reducción; Base de datos y rendimiento; Clasificación de objetos con KNN y KMeans. 

Se definio un agente racional que aprende, no omnisciente y que a partir de los algoritmos KNN y KMeans se entrene, aprenda  partir de sus percepciones 
y le permita ser autónomo gracias a la experiencia adquirida. Se utilizo el lenguaje de programación Python 2.7. 
Como objetivo se busca obtener un agente que en un entorno con condiciones controladas se desempeñe con un alto rendimiento.

Para ver todo el proceso del proyecto leer **Informe IA.pdf**
