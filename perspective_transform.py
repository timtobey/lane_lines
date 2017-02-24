#############################
# Loads raw image
# Undistorts image
# draws contour lines
# Saves warped image
########################


from Calibrate_Camera import *

# Load Camera data from Camera Calibration
objpoints = np.load("objpoints.npy")
imgpoints = np.load("imgpoints.npy")

# Read in an image
img = load_image('straight_lines1.jpg')
# Convert ot Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_size = (gray.shape[1], gray.shape[0])
# Undistort image
undistorted_image = cal_undistort(img, objpoints, imgpoints)

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
warped = cv2.warpPerspective(img, M, img_size)

# Draw warped photo with rectangle
warped_with_rectangle = cv2.rectangle(warped,
                                      (int(dst_top_left_corner[0]), int(dst_top_left_corner[1]))
                                      , (int(dst_bottom_right_corner[0]), int(dst_bottom_right_corner[1]))
                                      , (255, 0, 0), 3)

# Draw undistorted photo with polygon
poly = np.asarray(src, dtype=np.int)  # convert src to int
undistorted_image = cv2.polylines(undistorted_image, [poly], True, (255, 0, 0), 5)

compare_image(undistorted_image, warped_with_rectangle)

save_image("warped.jpg", warped)
