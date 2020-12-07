# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 5-1-2 Stereo Camera Calibration

# load library
import numpy as np
import cv2
import glob
import argparse
import pickle

# Defining the dimensions of checkerboard
a = 7
b = 6
square_size = 108 #mm

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, square_size, 0.1)
        #self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
        #                     cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((b*a, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + 'right/*.png')
        images_left = glob.glob(cal_path + 'left/*.png')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (a, b), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (a, b), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (a, b),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (a, b),
                                                  corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Ret', ret)
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print('Image Size', dims)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F),('imageSize', dims)])
        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
    # save the stereo  parameters
    with open('stereoParams.pickle', 'wb') as f:
        pickle.dump(cal_data, f)


# Intrinsic_mtx_1 – output first camera matrix
#
# dist_1 – output vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements. The output vector length depends on the flags.
#
# Intrinsic_mtx_2 – output second camera matrix
#
# dist_2 – output lens distortion coefficients for the second camera
#
# R – Output rotation matrix between the 1st and the 2nd camera coordinate systems.
#
# T – Output translation vector between the coordinate systems of the cameras.
#
# E – Output essential matrix.
#
# F – Output fundamental matrix.



# CV_CALIB_FIX_INTRINSIC: K and D matrices will be fixed. It is the default flag. If you calibrated your camera well, you can fix them so you’ll only get the rectification matrices.
# CV_CALIB_USE_INTRINSIC_GUESS: K and D matrices will be optimized. For this calculation, you should give well-calibrated matrices so that the result will be better(possibly).
# CV_CALIB_FIX_PRINCIPAL_POINT: Fix the reference point in the K matrix.
# CV_CALIB_FIX_FOCAL_LENGTH: Fix the focal length in the K matrix.
# CV_CALIB_FIX_ASPECT_RATIO: Fixing the aspect ratio.
# CV_CALIB_SAME_FOCAL_LENGTH: Calibrate the focal length and set Fx and Fy the same calibrated result. I am not familiar with this one but I am sure it’s required for specific stereo setups.
# CV_CALIB_ZERO_TANGENT_DIST: Remove the distortions.
# CV_CALIB_FIX_K1, …, CV_CALIB_FIX_K6: Remove distortion K1 to K6. Really important for experimentation. I am not familiar with the math behind those but the experiments I’ve made on these helped me a lot.
