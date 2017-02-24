# Try to change yellow to white
from PIL import Image, ImageMath
import numpy as np
import matplotlib.image as mpimg
import cv2
img = cv2.imread('test6.jpg', 1)


#img = mpimg.imread("test6.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#BGR order
#BLUE=0, GREEN=255, RED=255
blue_min_max = [0,100]
green_min_max = [100,255]
red_min_max = [100,255]
lower_range = np.array([blue_min_max[0],green_min_max[0],red_min_max[0]], dtype=np.uint8)
upper_range = np.array([blue_min_max[1],green_min_max[1],red_min_max[1]], dtype=np.uint8)


mask = cv2.inRange(hsv, lower_range, upper_range)
#res = cv2.bitwise_and(img,img,mask = mask)
# get first masked value (foreground)
fg = cv2.bitwise_or(img, img, mask=mask)

# get second masked value (background) mask must be inverted
mask = cv2.bitwise_not(mask)
background = np.full(img.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(background, background, mask=mask)
# combine foreground+background
final = cv2.bitwise_or(fg, bk)

cv2.imshow('mask', mask)
cv2.imshow('image', img)
cv2.imshow('Final', final)
#cv2.imshow('img2', img2)
while (1):
    k = cv2.waitKey(0)
    if (k == 27):
        break

cv2.destroyAllWindows()