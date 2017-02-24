from PIL import Image
import numpy as np
import cv2

#############################
#Filters Image to pull out white and yellow colors and makes them red
#############################
def threshold(image):
    im = np.asarray(image, dtype=np.uint8)
    #im = Image.fromarray(image)
    im = Image.fromarray(im)
    im = im.convert('RGBA')

    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace Yellow with red... (leaves alpha values alone...)
    yellow_areas = (red > 150) & (blue <140 ) & (green > 150)
    data[..., :-1][yellow_areas.T] = (255, 0, 0) # Transpose back needed

    # Replace white with red... (leaves alpha values alone...)
    white_areas = (red > 200) & (blue > 200) & (green > 200)
    data[..., :-1][white_areas.T] = (255, 0, 0) # Transpose back needed

    im2 = Image.fromarray(data)

    # make all pixels but red transparent
    datas = im2.getdata()
    newData = []
    for item in datas:
        if item[0] != 255 and item[1] != 0 and item[2] != 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    im2.putdata(newData)
    red_image = im2
    #im2.save("redtest.jpg")
    return red_image

###################################################
#Converts imgage to HLS and pulls out S channel
##################################################

def hls_select(image):
    thresh = (00, 255)
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result

    binary_output = binary  # placeholder line

    return binary_output

# threshold("test4.jpg")
