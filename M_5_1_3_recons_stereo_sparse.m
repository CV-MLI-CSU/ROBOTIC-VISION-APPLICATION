% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 
 
% Lecture 5-1-3 3D Reconstruction from Stereo Camera
close all; clear all; clc;
 
%% Read a Pair of Images
I1 = imread('images/cim0.png');
I2 = imread('images/cim1.png');
figure
imshowpair(I1, I2, 'montage');
title('Original Images');
 
%% Load Camera Parameters
load StereoCameraParameters.mat
 
%% Remove Lens Distortion
I1 = undistortImage(I1, stereoParams.CameraParameters1);
I2 = undistortImage(I2, stereoParams.CameraParameters2);
figure
imshowpair(I1, I2, 'montage');
title('Undistorted Images');
 
 
%% Find Point Correspondences Between The Images
% Detect feature points
imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'MinQuality', 0.1);
 
% Visualize detected points
figure
imshow(I1, 'InitialMagnification', 50);
title('150 Strongest Corners from the First Image');
hold on
plot(selectStrongest(imagePoints1, 150));
 
% Create the point tracker
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
 
% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, I1);
 
% Track the points
[imagePoints2, validIdx] = step(tracker, I2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);
 
% Visualize correspondences
figure
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
title('Tracked Features');
 
%% Estimate the Essential Matrix
% Estimate the fundamental matrix
[E, epipolarInliers] = estimateEssentialMatrix(...
    matchedPoints1, matchedPoints2, stereoParams.CameraParameters1, 'Confidence', 99.99);
 
% Find epipolar inliers
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);
 
% Display inlier matches
figure
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
title('Epipolar Inliers');
 
%% Compute the Camera Pose
[orient, loc] = relativeCameraPose(E, stereoParams.CameraParameters1, inlierPoints1, inlierPoints2);
 
%% Reconstruct the 3-D Locations of Matched Points
% Detect dense feature points. Use an ROI to exclude points close to the
% image edges.
roi = [30, 30, size(I1, 2) - 30, size(I1, 1) - 30];
imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'ROI', roi, ...
    'MinQuality', 0.001);
 
% Create the point tracker
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
 
% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, I1);
 
% Track the points
[imagePoints2, validIdx] = step(tracker, I2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);
 
% Compute the camera matrices for each position of the camera
% The first camera is at the origin looking along the Z-axis. Thus, its
% rotation matrix is identity, and its translation vector is 0.
camMatrix1 = cameraMatrix(stereoParams.CameraParameters1, eye(3), [0 0 0]);
 
% Compute extrinsics of the second camera
[R, t] = cameraPoseToExtrinsics(orient, loc);
camMatrix2 = cameraMatrix(stereoParams.CameraParameters1, R, t);
 
% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);
 
% Get the color of each reconstructed point
numPixels = size(I1, 1) * size(I1, 2);
allColors = reshape(I1, [numPixels, 3]);
colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);
 
% Create the point cloud
ptCloud = pointCloud(points3D, 'Color', color);
 
%% Display the 3-D Point Cloud
% Visualize the camera locations and orientations
cameraSize = 1.3;
figure
plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
hold on
grid on
plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, ...
    'Color', 'b', 'Label', '2', 'Opacity', 0);
 
% Visualize the point cloud
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
 
% Rotate and zoom the plot
camorbit(0, -30);
camzoom(4.5);

xlim([-50, 50]);
ylim([-50, 50]);
zlim([0,100]);
 
% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')
 
title('Up to Scale Reconstruction of the Scene');
 