% Load the video
video = VideoReader('video_file_name.video_format');

% Load the pretrained Shufflenet network
net = shufflenet;

% Create a face detector object
faceDetector = vision.CascadeObjectDetector;

% Initialize a counter for detected faces
faceCount = 0;

% Loop through each frame of the video
while hasFrame(video)
    % Read the current frame
    frame = readFrame(video);
    
    % Convert the current frame to grayscale
    grayFrame = rgb2gray(frame);
    
    % Detect faces in the current frame
    bbox = step(faceDetector, grayFrame);
    
    % Draw bounding boxes around the detected faces
    detectedFrame = insertObjectAnnotation(frame, 'rectangle', bbox, 'Face');
    
    % Display the current frame with bounding boxes
    imshow(detectedFrame);
    
    % Loop through each detected face
    for i = 1:size(bbox, 1)
        % Crop the detected face from the frame
        face = imcrop(frame, bbox(i, :));
        
        % Increment the face counter
        faceCount = faceCount + 1;

        % Save the detected face as an image in the same folder
        filename = sprintf('%d.png', faceCount);
        imwrite(face, filename);
    end
end

disp('Face detection and saving completed.');
