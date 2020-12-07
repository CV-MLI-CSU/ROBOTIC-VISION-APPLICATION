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
figure;
imshowpair(I1, I2, 'montage'); title('Original Images');
imshowpair(I1, I2); title('Original Images');

%% Load Stereo Camera Parameters
% Load the stereoParameters object.
load('StereoCameraParameters.mat');
% Visualize camera extrinsics.
showExtrinsics(stereoParams);

%% Rectify images
[frameLeftRect, frameRightRect] = ...
    rectifyStereoImages(I1, I2, stereoParams);
figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title('Rectified Images');

%% Compute Disparity
disparityMap = disparity(rgb2gray(frameLeftRect), rgb2gray(frameRightRect));
figure;
imshow(disparityMap, [0, 64]);
title('Disparity Map');
colormap jet
colorbar

%% Reconstruct the 3-D Scene
points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);