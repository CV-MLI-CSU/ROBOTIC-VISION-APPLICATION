% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 4-4-2 Region Properties

% load an image 
im = imread('images/cropped-sign.jpg');

% rgb --> gray
im_gray = rgb2gray(im);
imshowpair(im, im_gray, 'montage');

% binarization 
histogram(im_gray); 
bw = im2bw(im_gray, 0.39);
imshow(bw);

% check the region props.
stats = regionprops(bw,'all') ;

% filtering 
centerpoints = cat(1, stats.Centroid);
idx = find( centerpoints(:,1) > 129 & centerpoints(:,1) < 518 ...
    & centerpoints(:,2) > 81 & centerpoints(:,2) < 678); 

% removing undesirable objects
cc = bwconncomp(bw); 
bw2 = ismember(labelmatrix(cc), idx); 

% check the results
imshowpair(bw, bw2, 'montage');

