import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob , os

#os.chdir("C:/Users/timto/Google Drive/aaaa_udacity/Project_4/project_4_work/") # home
os.chdir("C:/Users/ttobey.GSI/Google Drive/aaaa_udacity/Project_4/project_4_work/") # Work

def calibrate_camera():
        curDir = os.getcwd()
        image_Path = os.path.join(curDir, "camera_cal/*jpg")
        # Arrays to store object points and image points from all the shapes
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in in image plane

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(7,5,0)
        objp = np.zeros((6*9,3),np.float32)
        objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)  # x, y coordinates

        for file in glob.glob(image_Path):
            img = mpimg.imread(file)

            # convert image to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # find corners of chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If corners are found, add object points, image points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        return imgpoints, objpoints

def cal_undistort(original_image, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs =\
        cv2.calibrateCamera(objpoints, imgpoints,
                            original_image.shape[0:2],None,None)
    undistorted_image = cv2.undistort(original_image, mtx, dist, None, mtx)
    return undistorted_image

def compare_image(original_image,undistorted_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted_image)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)

# show test image comparison
def show_test_image ():
    objpoints = np.load("objpoints.npy")
    imgpoints = np.load("imgpoints.npy")
    os.chdir("C:/Users/timto/Google Drive"
             "/aaaa_udacity/Project_4/project_4_work/camera_cal/")
    original_image = mpimg.imread("calibration1.jpg")
    undistorted_image = cal_undistort(original_image, objpoints, imgpoints)
    compare_image(original_image, undistorted_image)

#so I use the same load procedure for each image
def load_image(filename):
    img = mpimg.imread(filename)
    return img

def save_image(filename, file):
    mpimg.imsave(filename, file)
    print("Saved File: ",filename)



# Calculate and save imgpoints, objpoints
imgpoints, objpoints = calibrate_camera()

np.save("imgpoints", imgpoints)
np.save("objpoints", objpoints)