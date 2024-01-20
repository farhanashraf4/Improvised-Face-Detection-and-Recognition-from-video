function output_value = Load_database()
persistent loaded;
persistent numeric_Image;
if(isempty(loaded))
    all_Images = zeros(112*92, 10*10);
    for i=1:10
        folder_name = strcat('s', num2str(i));
        cd(folder_name);
        for j=1:10
            image_Container = imread(strcat(num2str(j),'.pgm'));
            % convert color image to grayscale
            if size(image_Container, 3) == 3
                image_Container = rgb2gray(image_Container);
            end
            % resize image to 112x92
            image_Container = imresize(image_Container, [112, 92]);
            all_Images(:,(i-1)*10+j)=reshape(image_Container,size(image_Container,1)*size(image_Container,2),1);
        end
        display('Loading Database');
        cd ..
    end
    numeric_Image = uint8(all_Images);
end
loaded = 1;
output_value = numeric_Image;
