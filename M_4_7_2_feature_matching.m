% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 4-7-2 Feature Matching

% read images 
%im1_rgb = imresize(imread('images/sign.jpg'), 0.35); 
im1_rgb = imresize(imread('images/cropped-sign.jpg'), 0.5); 
im2_rgb = imresize(imread('images/Dr1.JPG'), 0.35);
imshowpair(im1_rgb, im2_rgb, 'montage');
im1 = rgb2gray(im1_rgb);
im2 = rgb2gray(im2_rgb);


% Detect Feature Points / SURF
im1Points = detectSURFFeatures(im1);
im2Points = detectSURFFeatures(im2);

% Visualize the strongest feature points found in the im1.
figure;
imshow(im1_rgb);
title('100 Strongest Feature Points from im1');
hold on;
plot(selectStrongest(im1Points, 100));


% Visualize the strongest feature points found in the im2.
figure;
imshow(im2_rgb);
title('200 Strongest Feature Points from im2');
hold on;
plot(selectStrongest(im2Points, 200));

%  Extract Feature Descriptors - Extract feature descriptors at the interest points in both images.
[im1Features, im1Points] = extractFeatures(im1, im1Points);
[im2Features, im2Points] = extractFeatures(im2, im2Points);

% Find Putative Point Matches - Match the features using their descriptors.
pairs = matchFeatures(im1Features, im2Features);

% Display putatively matched features.
matchedim1Points = im1Points(pairs(:, 1), :);
matchedim2Points = im2Points(pairs(:, 2), :);
figure;
showMatchedFeatures(im1_rgb, im2_rgb, matchedim1Points, ...
    matchedim2Points, 'montage');
title('Putatively Matched Points (Including Outliers)');

%% Locate the Object in the im2 Using Putative Matches
% estimateGeometricTransform calculates the transformation relating ..
% the matched points, while eliminating outliers. This transformation ..
% allows us to localize the object in the im2.
[tform, inlierim1Points, inlierim2Points] = ...
    estimateGeometricTransform(matchedim1Points, matchedim2Points, 'affine');

% Display the matching point pairs with the outliers removed
figure;
showMatchedFeatures(im1_rgb, im2_rgb, inlierim1Points, ...
    inlierim2Points, 'montage');
title('Matched Points (Inliers Only)');

% switch im1 to % cropped-sign.jpg, run above code again
% Get the bounding polygon of the reference image.
im1Polygon = [1, 1;...                           % top-left
        size(im1, 2), 1;...                 % top-right
        size(im1, 2), size(im1, 1);... % bottom-right
        1, size(im1, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon
    
    
 % Transform the polygon into the coordinate system of the target image. ..
 % The transformed polygon indicates the location of the object in the im2.
newim1Polygon = transformPointsForward(tform, im1Polygon);

% Display the detected object.
figure;
imshow(im2_rgb);
hold on;
line(newim1Polygon(:, 1), newim1Polygon(:, 2), 'Color', 'y');
title('Detected Sign');


%% SIFT - vlfeat
run('vlfeat/toolbox/vl_setup')
vl_version verbose

% SIFT
close all, clear all, clc;
% prepare photos
im1_rgb = imresize(imread('images/cropped-sign.jpg'), 0.5); 
im2_rgb = imresize(imread('images/Dr1.JPG'), 0.35);
im1 = rgb2gray(im1_rgb);
im2 = rgb2gray(im2_rgb);
Ia = single(im1) ;
Ib = single(im2) ;

% compute the SIFT frames (keypoints) and descriptors 
[f,d] = vl_sift(Ia) ;
% visualize a random selection of 50 features
perm = randperm(size(f,2)) ;
sel = perm(1:50) ;
imshow(im1_rgb) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
% overlay the descriptors 
h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
set(h3,'color','g') ;

% Basic matching
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
[matches, scores] = vl_ubcmatch(da, db, 10) ;

% Display matched features.
[drop, perm] = sort(scores, 'descend');
sel = perm(1:50);

xa = fa(1, matches(1, sel));
xb = fb(1, matches(2, sel))+max(size(Ia,2),size(Ib,2));
ya = fa(2, matches(1, sel));
yb = fb(2, matches(2, sel));

figure(1);
imshowpair(im1_rgb, im2_rgb,  'montage');
hold on
h = line([xa; xb], [ya;yb]);
set(h, 'linewidth', 1, 'color', 'y');
vl_plotframe(fa(:,matches(1,sel))) ;
fb_new = fb;
fb_new(1,:) = fb(1, :) + max(size(Ia,2),size(Ib,2));
vl_plotframe(fb_new(:,matches(2,sel))) ;
hold off
