import cv2
from google.colab.patches import cv2_imshow

# Load the video
video = cv2.VideoCapture('video_file_name.video_format')

# Load the pretrained Shufflenet network (if needed)
# Note: You may need to find an equivalent pre-trained model in Python
# as Shufflenet is not available in OpenCV by default.

# Create a face detector object
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a counter for detected faces
face_count = 0

# Loop through each frame of the video
while video.isOpened():
    # Read the current frame
    ret, frame = video.read()
    
    if not ret:
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the detected face from the frame
        face = frame[y:y+h, x:x+w]

        # Increment the face counter
        face_count += 1

        # Save the detected face as an image in the same folder
        filename = f'{face_count}.png'
        cv2.imwrite(filename, face)

    # Display the current frame with bounding boxes
    cv2_imshow(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

print('Face detection and saving completed.')
