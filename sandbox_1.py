import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread("GOPR0050.jpg")

# Arrays to store object points and image points from all the shapes
objpoints = [] #3d points in real world space
imgpoints = [] # 2d points in in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(7,5,0)
objp = np.zeros((6*8,3),np.float32)
objp[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2) # x, y coordinates

#convert image to grayscale

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#find corensers of chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

# If corners are found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    #draw and dsiplay Corners

    img =cv2.drawChessboardCorners(img,(8,6),corners, ret)

plt.imshow(img)
plt.show(block=True)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

plt.imshow(dst)
plt.show(block=True)
