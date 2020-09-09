% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 3-6 Median Filter

I = rgb2gray(imread('lenna.png'));
J = imnoise(I,'salt & pepper',0.02);
imshow(J)
im = imfilter(J, ones(3,3)/9); imshow(im);
im1 = imfilter(J, ones(5,5)/25); imshow(im1);
I2 = medfilt2(J ); imshow(I2);
