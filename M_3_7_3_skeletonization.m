% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 3-7-3 Skeletonization

% load an image & turn it to gray
img = rgb2gray(imread('images/railway.jpg')); 

% binarization
bw = im2bw(img);
imshowpair(img, bw, 'montage');

% morphological operations
se = strel('disk', 3);
bw2 = imopen(bw, se);
bw3 = bwareaopen(bw2, 5000);
se2 = strel('disk', 10);
bw4 = imdilate(bw3, se2);
img2 = [bw, bw2;bw3, bw4];
imshow(img2);

% skeletonization
bw = bwskel(bw4);
imshow([bw4,bw]);

% find the intersection
out = bwmorph(bw,'branchpoints'); 
imshow([bw, out]);
out2 = imdilate(out,se2);
imshow([bw, out, out2]);
out(:, 1:4) = 1;
out2(:, 1:4) = 1;
imshow([bw, out, out2]);


