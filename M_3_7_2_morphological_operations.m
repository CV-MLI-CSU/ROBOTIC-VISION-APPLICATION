% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 3-7-2 Morphological Operations

% load an image & turn it to gray
img = rgb2gray(imread('images/chips.png')); 
imshow(img); 
bw = im2bw(img);imshow(bw); 

% define a morphological structuring element
se1 = strel('square',5);
se2 = strel('disk',5);
se3 = strel('line',10,30);

% ERODE
im1 = imerode(bw, se1); figure(1), imshow(im1);
im2 = imerode(bw, se2); figure(2), imshow(im2);
im3 = imerode(bw, se3); figure(3), imshow(im3);

% open and close operation 
im4 = imopen(bw, se1);
im5 = imclose(bw, se1);
figure(3), imshowpair(im4, im5, 'montage');





