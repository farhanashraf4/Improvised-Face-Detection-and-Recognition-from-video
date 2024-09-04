import cv2
from mtcnn import MTCNN
from google.colab.patches import cv2_imshow
import time

# Load the video
video = cv2.VideoCapture('video_file_name.video_format')

# Load pretrained shufflenet model (if needed)

# Create MTCNN face detector
detector = MTCNN()

# Create a counter to keep track of detected faces
face_count = 0

# Get the start time
start_time = time.time()

# Loop through each frame of the video
while video.isOpened():
    # Read the current frame
    ret, frame = video.read()

    if not ret:
        break

    # Detect faces in the current frame using MTCNN
    results = detector.detect_faces(frame)

    # Increment the face counter
    face_count += len(results)

    # Draw bounding boxes around the detected faces
    for result in results:
        bounding_box = result['box']
        score = result['confidence']
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (255, 0, 0), 2)
        cv2.putText(frame, f'Score: {score:.2f}', (bounding_box[0], bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Crop the detected face from the frame
        face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                     bounding_box[0]:bounding_box[0] + bounding_box[2]]

        # Increment the face counter
        face_count += 1

        # Save the detected face to a file
        filename = f'{face_count}.pgm'
        cv2.imwrite(filename, face)

    # Display the frame
    cv2_imshow(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Get the end time
elapsed_time = time.time() - start_time

# Release the video capture object
video.release()

# Print the total number of detected faces
print(f'Total Detected Faces: {face_count}')

# Print the execution time
print(f'Execution Time: {elapsed_time:.3f} seconds')
