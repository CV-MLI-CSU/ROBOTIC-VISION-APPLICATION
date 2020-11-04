micheal = imread('images/train.png');
imshow(img);
red  = img(:, :, 1);
g = img(:, :, 2);
b = im(:, :, 3);
imshowpair(red, g, 'montage');