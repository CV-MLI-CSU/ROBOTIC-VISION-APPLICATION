% Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> 
% Uni: Central South University, Changsha, China 
% Online course: Robotic Vision and Applications 
% Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0 
% Copyright © 2020 SR2V Ltd. All rights reserved 

% Lecture 3-5 template matching 
im = imread('images/dinner.jpg');
im2 = imread('images/danielface.jpg');
imshowpair(im, im2, 'montage');
[I_SSD,I_NCC,Idata]=template_matching(im2,im);  
imshowpair(I_SSD, I_NCC, 'montage');
[M,I] = max(I_NCC(:));
[I_row, I_col] = ind2sub(size(I_NCC),I);
imshow(im), hold on, 
plot(I_col, I_row, 'bo', 'MarkerSize',120, 'LineWidth',3);
plot(I_col, I_row, 'g*', 'MarkerSize',15, 'LineWidth',3);
hold off
