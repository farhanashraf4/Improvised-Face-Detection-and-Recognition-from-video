% Load the test images
all_Test_Images = zeros(112*92, 10*10);
for i=1:10
    cd(strcat('s',num2str(i+10)));
    for j=1:10
        image_Container = imread(strcat(num2str(j),'.pgm'));
        % convert color image to grayscale
        if size(image_Container, 3) == 3
            image_Container = rgb2gray(image_Container);
        end
        % resize image to 112x92
        image_Container = imresize(image_Container, [112, 92]);
        all_Test_Images(:,(i-1)*10+j)=reshape(image_Container,size(image_Container,1)*size(image_Container,2),1);
    end
    display('Loading Test Images');
    cd ..
end

% Load the database
loaded_Image = Load_database();

% Run the face recognition method on the test images
num_Test_Images = size(all_Test_Images, 2);
correctly_Recognized = 0;
for i = 1:num_Test_Images
    test_Image = all_Test_Images(:,i);
    % Find the closest image in the database
    p = test_Image - mean_value;
    s = single(p)'*V;
    z = [];
    for j = 1:size(rest_of_the_images, 2)
        z = [z,norm(all_image_Signatire(j,:)-s,2)];
    end
    [a, recognized_Index] = min(z);
    % Check if the recognized identity matches the ground truth
    if ceil(recognized_Index/10) == i+10
        correctly_Recognized = correctly_Recognized + 1;
    end
end

% Calculate the accuracy
accuracy = correctly_Recognized / num_Test_Images;
fprintf('Accuracy: %.2f%%\n', accuracy*100);
