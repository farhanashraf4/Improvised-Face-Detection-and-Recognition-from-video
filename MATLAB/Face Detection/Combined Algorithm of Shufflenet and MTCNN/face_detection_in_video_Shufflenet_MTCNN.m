% 1. Load the video
video = VideoReader('video_file_name.video_format');

% 2. Load pretrained shufflenet model
net = shufflenet;

% 3. Create a face detector object
detector = mtcnn.Detector();

% 4. Create a counter to keep track of detected faces
faceCount = 0;

% 5. Get the start time
tic;

% 6. Loop through each frame of the video
while hasFrame(video)
    % 7. Read the current frame
    frame = readFrame(video);

    % 8. Detect faces in the current frame
    [bboxes, scores, ~] = detector.detect(frame);
    % Increment the face counter
    faceCount = faceCount + numel(scores);

    % 9. Draw bounding boxes around the detected faces
    displayIm = insertObjectAnnotation(frame, 'rectangle', bboxes, scores, 'LineWidth', 2);
    imshow(displayIm);

    % 10. Loop through each detected face
    for i = 1:size(bboxes, 1)
        % Crop the detected face from the frame
        face = imcrop(frame, bboxes(i, :));
        % Increment the face counter
        faceCount = faceCount + 1;

        % 10. Save the detected face to a file
        filename = sprintf('%d.pgm', faceCount);
        imwrite(face, filename);
    end  % End of the loop for processing each detected face

end  % End of the loop for processing each frame

% 11. Get the end time
elapsedTime = toc;

% 12. Print the total number of detected faces
fprintf('Total Detected Faces: %d\n', faceCount);

% 13. Print the execution time
fprintf('Execution Time: %.3f seconds\n', elapsedTime);
