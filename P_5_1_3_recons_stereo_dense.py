# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 5-1-2 Stereo Camera Calibration
# Simple example of stereo image matching and point cloud generation.
# Resulting .ply file cam be easily viewed using MeshLab (
# http://meshlab.sourceforge.net/ )

# load library
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from P_5_1_2_stereo_camera_calibration import *
import os

leftFrame = cv2.imread('images/camera_calib/left/left01.png')
rightFrame = cv2.imread('images/camera_calib/right/right01.png')
origin_pair = np.concatenate((leftFrame, rightFrame), axis=1) #side by side comparison
cv2.imshow('origin_pair', cv2.resize(origin_pair,None,fx=0.5,fy=0.5)) # Display img
cv2.waitKey(0)

# load the stereo  parameters: Ret, Intrinsic_mtx_1&2, dist_1&2, R, T, E, F, Image Size,
with open('stereoParams.pickle', 'rb') as f:
    stereoParams = pickle.load(f)
    #print(stereoParams.camera_model)
locals().update(stereoParams.camera_model)

# Undistortion and Rectification part!
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(M1, dist1, M2, dist2,
    imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
# # alpha=-1 -> Let OpenCV optimize black parts.
# # alpha= 0 -> Rotate and cut the image so that there will be no black parts. This option cuts the
# #  image so badly most of the time, that you won’t have a decent high-quality image but worth to try.
# # alpha= 1 -> Make the transform but don’t cut anything.
# # alpha=experimental-> Sometimes nothing works for you. This means that you should experiment
# # with the values. If it’s okay for you to have some black part, but the high quality image,
# # you can work on the alpha value. I found the best at 0.9756 for my camera, so don’t lose hope :)

height, width, channel = leftFrame.shape  # We will use the shape for remap
leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, dist1, R1, P1, (width, height), cv2.CV_32FC1)
left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, dist2, R2, P2, (width, height), cv2.CV_32FC1)
right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

rectified_pair = np.concatenate((left_rectified, right_rectified), axis=1) #side by side comparison
cv2.imshow('rectified_pair', cv2.resize(rectified_pair,None,fx=0.5,fy=0.5)) # Display img
cv2.waitKey(0)

# # We need grayscale for disparity map.
gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# calculate disparity by cv2.StereoBM_create
stereo = cv2.StereoBM_create(numDisparities=112, blockSize=11)
disparity = stereo.compute(gray_left,gray_right)
plt.imshow(disparity,'gray')
plt.show()

# ---------- generate paired points -----------

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
       f.write(ply_header % dict(vert_num=len(verts)))
       np.savetxt(f, verts, '%f %f %f %d %d %d')

def remove_invalid(disp_arr, points, colors):
    mask = (
        (disp_arr > disp_arr.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
    return points[mask], colors[mask]

if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.imread('images/im0.png')
    imgR = cv2.imread('images/im1.png')
    rectified_pair = np.concatenate((imgL, imgR), axis=1) #side by side comparison
    cv2.imshow('rectified_pair', cv2.resize(rectified_pair,None,fx=0.5,fy=0.5)) # Display img
    cv2.waitKey(0)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 128-min_disp
    stereo = cv2.StereoSGBM_create(
    	blockSize = 5,
    	numDisparities = num_disp,
    	minDisparity = min_disp,
    	P1 = 8*3*window_size**2,
    	P2 = 32*3*window_size**2,
    	disp12MaxDiff = 1,
    	uniquenessRatio = 15,
    	speckleWindowSize = 0,
    	speckleRange = 5,
    	preFilterCap = 63,
    	mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    	)

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...')
    h, w = imgL.shape[:2]
#    tx = 80 # distance_between_cameras / baseline
    f = 0.8*w                          # guess for focal length
#    f = M1[0,0]
    Q = np.float32([[1, 0, 0, -0.5*w],
                   [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                   [0, 0, 0,     -f], # so that y-axis looks up
        #           [0, 0, 1/tx,      0]])
                   [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    print(disp.min())
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    if os.path.exists(out_fn):
        os.remove(out_fn)
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
