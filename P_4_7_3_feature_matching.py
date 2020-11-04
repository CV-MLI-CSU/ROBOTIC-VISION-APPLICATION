# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 4-7-3 Feature Matching

# load library
import numpy as np
import cv2

# read images
img1_rgb = cv2.resize(cv2.imread('images/cropped-sign.jpg'),None,fx=0.5,fy=0.5) # queryImage
img2_rgb = cv2.resize(cv2.imread('images/Dr1.JPG'), None,fx=0.25,fy=0.25) # trainImage
img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)

# CHECK DIFFERENT FEATURES
# F1. -------- Haris corner detection --------
img2_F1 = img2_rgb.copy()
img2_32 = np.float32(img2)
dst = cv2.cornerHarris(img2_32, blockSize=2, ksize=3, k=0.04)
# dilate to mark the corners
dst = cv2.dilate(dst, None)
img2_F1[dst > 0.05 * dst.max()] = [0, 255, 0]
# show feature points
cv2.imshow('haris_corner', img2_F1)
cv2.waitKey()

# F2. --------  Shi-Tomasi corner detection --------
img2_F2 = img2_rgb.copy()
corners = cv2.goodFeaturesToTrack(img2, maxCorners=50, qualityLevel=0.02, minDistance=10)
corners = np.float32(corners)
for item in corners:
    x, y = item[0]
    cv2.circle(img2_F2, (x, y), 6, (0, 255, 0), -1)
# show feature points
cv2.imshow('Shi-Tomasi_corner', img2_F2)
cv2.waitKey()

# # F3. --------  SIFT (Scale-Invariant Feature Transform) --------
# img2_F3 = img2_rgb.copy()
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(img2, None)
# kp_img = cv2.drawKeypoints(img2_F3, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # show feature points
# cv2.imshow('SIFT', kp_img)
# cv2.waitKey()
#
# # F4. --------  SURF (Speeded-Up Robust Features) --------
# img2_F4 = img2_rgb.copy()
# surf = cv2.xfeatures2d.SURF_create()
# kp, des = surf.detectAndCompute(img2, None)
# kp_img = cv2.drawKeypoints(img2_F4, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # show feature points
# cv2.imshow('SURF', kp_img)
# cv2.waitKey()

# F5. --------  FAST algorithm for corner detection --------
img2_F5 = img2_rgb.copy()
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(True)
fast.setThreshold(50) # default is 1, too many key points...
kp = fast.detect(img2, None)
kp_img = cv2.drawKeypoints(img2_F5, kp, None, color=(0, 255, 0))
# show feature points
cv2.imshow('FAST', kp_img)
cv2.waitKey()
cv2.destroyAllWindows # close the img window

# F6. --------  ORB --------
img2_F6 = img2_rgb.copy()
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(img2, None)
kp_img = cv2.drawKeypoints(img2_F6, kp, None, color=(0, 255, 0), flags=0)
# show feature points
cv2.imshow('ORB', kp_img)
cv2.waitKey()
cv2.destroyAllWindows # close the img window


## FEATURE Matching
# find the keypoints and descriptors with SIFT
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# M1. --------------  Brute-Force (BF) Matcher ------------------
# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# draw first 50 matches
match_img_BF = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches[:50], None)
cv2.imshow('Brute-Force (BF) Matches', match_img_BF)
cv2.waitKey()

# M2. --------------  FLANN based matcher ------------------
# FLANN parameters
index_params = dict(algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=2)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# As per Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
# draw all good matches
match_img_FLANN = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None)
cv2.imshow('FLANN based Matches', match_img_FLANN)
cv2.waitKey()
