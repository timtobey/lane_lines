import numpy as np
import matplotlib.pyplot as plt
from Calibrate_Camera import load_image, save_image, compare_image
from filter_image_and_hls import threshold, hls_select
import matplotlib.image as mpimg
import cv2
#img1 = threshold("warped.jpg")
#img1=load_image("warped.jpg")
import os
os.chdir("C:/Users/ttobey.GSI/Google Drive/aaaa_udacity/Project_4/project_4_work/")

img2 = threshold("warped.jpg")
img3 = hls_select(img2)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img2)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(img3, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show(block=True)