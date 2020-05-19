import numpy as np
import matplotlib.pyplot as plt

from skimage import filters, io, color

prueba = './ejemplos/tornillo_prueba.jpg'
image = io.imread(prueba)
image = color.rgb2gray(image)

edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

sobel_v=filters.sobel_v(image)
sobel_h=filters.sobel_h(image)

fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Operador cruzado de Robert')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Operador de Sobel')

axes[2].imshow(sobel_v, cmap=plt.cm.gray)
axes[2].set_title('Operador de Sobel vertical')

axes[3].imshow(sobel_h, cmap=plt.cm.gray)
axes[3].set_title('Operador de Sobel horizontal')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

#Bibliografia consultada
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_edge_filter.html#sphx-glr-auto-examples-edges-plot-edge-filter-py
