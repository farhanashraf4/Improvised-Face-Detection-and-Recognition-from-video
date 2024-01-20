% Read the video file
video = VideoReader('video_file_name.video_format');

% Create MTCNN face detector
detector = mtcnn.Detector();

% Initialize face counter
faceCount = 0;

% Start measuring execution time
tic;

% Process each frame in the video
while hasFrame(video)
    % Read a frame from the video
    frame = readFrame(video);

    % Detect faces in the frame using MTCNN
    [bboxes, scores, ~] = detector.detect(frame);

    % Increment the face counter based on the number of detected faces
    faceCount = faceCount + numel(scores);

    % Display the frame with bounding boxes and scores
    displayIm = insertObjectAnnotation(frame, 'rectangle', bboxes, scores, 'LineWidth', 2);
    imshow(displayIm);

    % Process each detected face
    for i = 1:size(bboxes, 1)
        % Crop the face from the frame
        face = imcrop(frame, bboxes(i, :));

        % Increment the face counter
        faceCount = faceCount + 1;

        % Generate a filename for the face image
        filename = sprintf('%d.pgm', faceCount);

        % Save the face image
        imwrite(face, filename);
    end  % End of the loop for processing each face

end  % End of the loop for processing each frame

% Stop measuring execution time
elapsedTime = toc;

% Display total number of detected faces and execution time
fprintf('Total Detected Faces: %d\n', faceCount);
fprintf('Execution Time: %.3f seconds\n', elapsedTime);
