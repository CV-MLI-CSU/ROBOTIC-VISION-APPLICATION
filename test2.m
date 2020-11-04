img = imread(imageFileNames{1});
imshow(img); hold on
plot(imagePoints(:,1,1),imagePoints(:,2,1),'db');
hold off