import cv2
from mtcnn import MTCNN
from google.colab.patches import cv2_imshow
import time

# Load the video
video = cv2.VideoCapture('ideo_file_name.video_format')

# Create MTCNN face detector
detector = MTCNN()

# Initialize face counter
face_count = 0

# Start measuring execution time
start_time = time.time()

# Process each frame in the video
while video.isOpened():
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break

    # Detect faces in the frame using MTCNN
    results = detector.detect_faces(frame)

    # Increment the face counter based on the number of detected faces
    face_count += len(results)

    # Display the frame with bounding boxes and scores
    for result in results:
        bounding_box = result['box']
        score = result['confidence']
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),
                      (255, 0, 0), 2)
        cv2.putText(frame, f'Score: {score:.2f}', (bounding_box[0], bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Crop the face from the frame
        face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                     bounding_box[0]:bounding_box[0] + bounding_box[2]]

        # Increment the face counter
        face_count += 1

        # Generate a filename for the face image
        filename = f'{face_count}.pgm'

        # Save the face image
        cv2.imwrite(filename, face)

    # Display the frame
    cv2_imshow(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop measuring execution time
elapsed_time = time.time() - start_time

# Release the video capture object
video.release()

# Display total number of detected faces and execution time
print(f'Total Detected Faces: {face_count}')
print(f'Execution Time: {elapsed_time:.3f} seconds')
