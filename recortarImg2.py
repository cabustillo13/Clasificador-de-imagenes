#import cv2
#import numpy as np

#im = cv2.imread("Data Base/YTrain/ZArandelas/photo0.jpg")
#gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(gray,127,255,0)
#bordes = cv2.Canny(gray, 100, 200)
#Para OpenCV4
#contours, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#areas = [cv2.contourArea(c) for c in contours]
#max_index = np.argmax(areas)
#cnt=contours[max_index]
#x,y,w,h = cv2.boundingRect(cnt)

#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow("Show",im)
#cv2.imwrite("crack.jpg", im)
#cv2.waitKey(0)

#################################################################################################################
import numpy as np
import cv2

def crop_min_rect(img, countour):
    rect = cv2.minAreaRect(countour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w, h = (int(n) for n in  rect[1])
    xs, ys = zip(*box)
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    center = int((x1 + x2) / 2), int((y1 + y2) / 2)
    size = int((x2 - x1)), int((y2 - y1))

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    rot_m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    cropped = cv2.warpAffine(cv2.getRectSubPix(img, size, center), rot_m, size)

    cw = w if not rotated else h
    ch = h if not rotated else w

    img_crop = cv2.getRectSubPix(cropped, (cw, ch), (size[0] / 2, size[1] / 2))
    im_draw  = cv2.drawContours(img, [box], -1, (0, 0, 255), 5)
    return im_draw, img_crop

KERNEL_SIZE = 90

img = cv2.imread('image12.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 64, 255, 0)

kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
img_e = cv2.dilate(thresh, kernel, iterations=1)
contours, _ = cv2.findContours(img_e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_countour = max(contours, key = cv2.contourArea)

cont_img, crop_img = crop_min_rect(img, max_countour)

cv2.imwrite("countour.png", cont_img)
cv2.imwrite("cropped.png", crop_img)
#############################################################################################################

