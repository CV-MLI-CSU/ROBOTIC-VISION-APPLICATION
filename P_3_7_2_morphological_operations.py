# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright © 2020 SR2V Ltd. All rights reserved

# Lecture 3-7-2 Morphological Operations

# import lib
import cv2
import numpy as np

# Load image a RGB image and convert it to GRAYONE
img = cv2.imread('images/chips.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,None,fx=0.7,fy=0.7) # resize the image by 70%

# binarization
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# show images
compare1 = np.concatenate((img, th2), axis=1) #side by side comparison
cv2.imshow('compare1', compare1) # Display img

# define a morphological structuring element
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # Rectangular / Square Kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # disk / ellipse
kernel3 = np.zeros((10,10),np.uint8) # line, degree = 30
kernel3[9,0] = 1
kernel3[8,2] = 1
kernel3[7,4] = 1
kernel3[6,6] = 1
kernel3[5,8] = 1

# ERODE
im1 = cv2.erode(th2,kernel1,iterations = 1)
im2 = cv2.erode(th2,kernel2,iterations = 1)
im3 = cv2.erode(th2,kernel3,iterations = 1)

# show images
compare2 = np.concatenate((im1, im2, im3), axis=1) #side by side comparison
cv2.imshow('compare2', compare2) # Display img

cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows # close the img window


# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
