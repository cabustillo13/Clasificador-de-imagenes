import cv2
import numpy as np

im = cv2.imread("Data Base/YTrain/ZArandelas/photo0.jpg")
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
bordes = cv2.Canny(gray, 100, 200)
#Para OpenCV4
contours, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]
x,y,w,h = cv2.boundingRect(cnt)

cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow("Show",im)
cv2.imwrite("crack.jpg", im)
cv2.waitKey(0)
