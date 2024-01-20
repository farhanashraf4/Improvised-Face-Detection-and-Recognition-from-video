image_Container = imread(strcat(num2str(j),'.pgm'));
if size(image_Container, 3) == 3 % check if the image is not grayscale
    image_Container = rgb2gray(image_Container);
end
image_size = size(image_Container);
if ~(image_size(1) == 112 && image_size(2) == 92)
    error('Error: Image size is not 112x92.');
end
