# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright © 2020 SR2V Ltd. All rights reserved

# Lecture 3-6 Median Filter

# import lib
import cv2
import numpy as np

# Load image a RGB image and convert it to GRAYONE
img = cv2.imread('images/lenna.png', cv2.IMREAD_GRAYSCALE)

# add noise (salt & paper)
# Generate Gaussian noise
gauss = np.random.normal(0,0.5,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
# Add the Gaussian noise to the image
img_noise = img + img * gauss

img_gauss = cv2.GaussianBlur(img_noise, (5,5), 0) # guassian blur

img_median = cv2.medianBlur(img_noise, 5) # Add median filter to image

compare1 = np.concatenate((img, img_noise), axis=1) #side by side comparison
compare2 = np.concatenate((img_gauss, img_median), axis=1) #side by side comparison
compare = np.concatenate((compare1,compare2), axis=0) #side by side comparison

cv2.imshow('img', compare) # Display img with median filter
cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows # close the img window
