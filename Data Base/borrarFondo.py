import numpy as np
import cv2

#Load the Image
imgo = cv2.imread('/home/carlos/Documentos/Image-Classifier/Data Base/YTrain/ZTornillos/photo0.jpg')
height, width = imgo.shape[:2]

#Create a mask holder
mask = np.zeros(imgo.shape[:2],np.uint8)

#Grab Cut the object
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#Hard Coding the Rect The object must lie within this rect.
rect = (1079,1079,width-1,height-1)
cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = imgo*mask[:,:,np.newaxis]

#Get the background
background = imgo - img1

#Change all pixels in the background that are not black to white
background[np.where((background > [100,100,100]).all(axis = 2))] = [255,255,255]


#Add the background and the image
final = background + img1

#To be done - Smoothening the edges

cv2.imwrite('image12.jpg', final )

cv2.waitKey(0)
cv2.destroyAllWindows()
 
