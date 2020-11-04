% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 4-2-2 Camera Calibration

% Create a set of calibration images.
path = 'images/camera_calib/';
images = imageDatastore(path);
imageFileNames = images.Files;

% Detect calibration pattern.
[imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);

% Generate world coordinates of the corners of the squares.
squareSize = 30; % millimeters
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera.
I = readimage(images, 1); 
imageSize = [size(I, 1), size(I, 2)];
[cameraParams, ~, estimationErrors] = estimateCameraParameters(imagePoints, ...
    worldPoints, 'ImageSize', imageSize);

                                 
figure; 
showExtrinsics(cameraParams, 'CameraCentric');
figure; 
showExtrinsics(cameraParams, 'PatternCentric');

displayErrors(estimationErrors, cameraParams);

% Remove lens distortion and display results.
I = imread('images/woodblockyellow.jpg'); imshow(I);
J1 = undistortImage(I,cameraParams);
figure; imshowpair(I,J1,'montage');
figure; imshowpair(I,J1);









