#############################
# Loads raw image
# Undistorts image
# draws contour lines
# Saves warped image
########################


from Calibrate_Camera import *

def load_calbaration_points():
    # Load Camera data from Camera Calibration
    objpoints = np.load("objpoints.npy")
    imgpoints = np.load("imgpoints.npy")
    return objpoints, imgpoints

def undistorted_image(image, objpoints, imgpoints):
    # Undistort image
    undistorted_image = cal_undistort(image, objpoints, imgpoints)
    return undistorted_image

def warp_my_image_mr_scotty(undistorted_image):
    img = undistorted_image
    # Convert ot Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    # select area of source image
    offset = 200  # offset for dst points

    src_top_right_corner = [(img_size[0] / 2) - 55, img_size[1] / 2 + 100]
    src_bottom_right_corner = [(img_size[0] / 6 - 10), img_size[1]]
    src_bottom_left_corner = [(img_size[0] * 5 / 6) + 60, img_size[1]]
    src_top_left_corner = [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]

    src = np.float32(
        [[src_top_right_corner[0], src_top_right_corner[1]],
         [src_bottom_right_corner[0], src_bottom_right_corner[1]],
         [src_bottom_left_corner[0], src_bottom_left_corner[1]],
         [src_top_left_corner[0], src_top_left_corner[1]]])

    # select area of destination image
    dst_top_right_corner = [(img_size[0] / 4), 0]
    dst_bottom_right_corner = [(img_size[0] / 4), img_size[1]]
    dst_bottom_left_corner = [(img_size[0] * 3 / 4), img_size[1]]
    dst_top_left_corner = [(img_size[0] * 3 / 4), 0]

    dst = np.float32(
        [[dst_top_right_corner[0], dst_top_right_corner[1]],
         [dst_bottom_right_corner[0], dst_bottom_right_corner[1]],
         [dst_bottom_left_corner[0], dst_bottom_left_corner[1]],
         [dst_top_left_corner[0], dst_top_left_corner[1]]])

    # Perform Perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, M, img_size)
    return warped_image ,dst, src

def warped_image_with_rectangle(warped_image, dst):

    # Draw warped photo with rectangle
    dst = np.reshape(dst,(1,-1))
    dst = dst.squeeze()
    warped_image_with_rectangle = cv2.rectangle(warped_image,
                                          (dst[6],dst[7])
                                          ,(dst[2], dst[3])
                                          , (255, 0, 0), 3)
    return warped_image_with_rectangle

def undistorted_image_with_poly(undistorted_image,src):
    # Draw undistorted photo with polygon
    poly = np.asarray(src, dtype=np.int)  # convert src to int
    undistorted_image_with_poly = cv2.polylines(undistorted_image, [poly], True, (255, 0, 0), 5)
    return undistorted_image_with_poly
#compare_image(undistorted_image_with_poly, warped_with_rectangle)

#save_image("warped.jpg", warped)
