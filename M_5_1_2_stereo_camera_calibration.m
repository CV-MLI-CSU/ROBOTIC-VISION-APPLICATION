% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 5-1-2 Stereo Camera Calibration

close all; clear all; clc;

% read images 
leftImages = imageDatastore('images/camera_calib/left/');
rightImages = imageDatastore('images/camera_calib/right/');

% Detect the checkerboards.
[imagePoints,boardSize] = ...
  detectCheckerboardPoints(leftImages.Files,rightImages.Files);

% Specify the world coordinates of the checkerboard keypoints. Square size is in millimeters.
squareSize = 108;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

% Calibrate the stereo camera system. Both cameras have the same resolution.
I = readimage(leftImages,1); 
imageSize = [size(I,1),size(I,2)];
params = estimateCameraParameters(imagePoints,worldPoints, ...
                                  'ImageSize',imageSize);

% Visualize the calibration accuracy.
showReprojectionErrors(params);

% Visualize camera extrinsics.
figure;
showExtrinsics(params);

% save parameters
save('StereoCameraParameter.mat','params');
