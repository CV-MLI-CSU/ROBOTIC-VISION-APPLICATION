# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright © 2020 SR2V Ltd. All rights reserved

# Lecture 3-7-3 Skeletonization

# import lib
import cv2
import numpy as np
from getSkeletonIntersection import getSkeletonIntersection

# load an image & turn it to gray
img = cv2.imread('images/railway.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,None,fx=0.5,fy=0.5) # resize the image by 70%
cv2.imshow('img', img) # Display img

# thresholding
bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# show images
cv2.imshow('bw', bw) # Display img

# morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # disk / ellipse
bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

# Filter using contour area and remove small noise
bw3 = bw2.copy() # copy bw2 to bw3
cnts = cv2.findContours(bw2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 3000:
        cv2.drawContours(bw2, [c], -1, (0,0,0), -1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)) # disk / ellipse
bw4 = cv2.dilate(bw3, kernel2);
compare1 = np.concatenate((bw, bw2), axis=1) #side by side comparison
compare2 = np.concatenate((bw3, bw4), axis=1) #side by side comparison
img2 = np.concatenate((compare1,compare2), axis=0) #side by side comparison
cv2.imshow('img2', img2) # Display img

#  skeletonization
bw = cv2.ximgproc.thinning(bw4)
compare1 = np.concatenate((bw4, bw), axis=1) #side by side comparison
cv2.imshow('compare1', compare1) # Display img

# find the intersection
points = getSkeletonIntersection(bw)
centroids = np.array(points)
radius = 5 # Radius of circle
color = (0, 0, 255) # Red color in BGR
thickness = -1 # Line thickness of -1 px
# Using cv2.circle() method, draw a circle of red color of thickness -1 px
out = cv2.merge([bw,bw,bw])
for point in centroids:
    cv2.circle(out,(tuple(point)),radius,(0,0,255),thickness)
# show results
cv2.imshow('out', out) # Display img

cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows # close the img window
