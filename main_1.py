from Calibrate_Camera import compare_image
from perspective_transform_function import load_calbaration_points, warp_my_image_mr_scotty,\
    warped_image_with_rectangle,undistorted_image_with_poly, undistorted_image
import cv2, numpy as np
from filter_image_and_hls import threshold, hls_select
from find_lane_lines import window_and_fit


original_image = cv2.imread("straight_lines1.jpg",1)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

objpoints, imgpoints =load_calbaration_points() # load pints from calibration
undistorted_image = undistorted_image(original_image,objpoints,imgpoints)
warped_image , dst, src  = warp_my_image_mr_scotty(undistorted_image)# create undistorted and warped image
warped_image_with_rectangle = warped_image_with_rectangle(warped_image, dst) # draw on warped image
undistorted_image_with_poly = undistorted_image_with_poly(undistorted_image,src) # draw poly on undistorted image
compare_image(undistorted_image_with_poly,warped_image_with_rectangle,)# compare images
red_image = threshold(warped_image) # make white and yellow colors red
binary_warped = hls_select(red_image)# convert to hls
left_fit, right_fit, left_fitx, right_fitx, ploty  = window_and_fit(binary_warped) # find lane lines


