import numpy as np
import cv2
import matplotlib.pyplot as plt

loaded = 0  # Initialize the loaded variable
numeric_image = None

def load_database():
    global loaded, numeric_image
    if not loaded:
        all_images = np.zeros((112 * 92, 10 * 10), dtype=np.uint8)
        for i in range(1, 11):
            folder_name = f's{i}'
            for j in range(1, 11):
                image_path = f'{folder_name}/{j}.pgm'
                image_container = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image_container is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image_container = cv2.resize(image_container, (92, 112))

                all_images[:, (i - 1) * 10 + j - 1] = image_container.flatten()
            print('Loading Database')
        numeric_image = all_images
    loaded = 1
    return numeric_image

# Initialize loaded before calling load_database
loaded_image = load_database()

# Eigenface recognition
image_signature = 20
random_index = int(100 * np.random.rand())
random_image = loaded_image[:, random_index]
rest_of_the_images = np.delete(loaded_image, random_index, axis=1)

white_image = np.ones((1, rest_of_the_images.shape[1]), dtype=np.uint8)
mean_value = np.mean(rest_of_the_images, axis=1, dtype=np.uint8)
mean_removed = rest_of_the_images - np.outer(mean_value, white_image)

L = np.dot(mean_removed.T, mean_removed)
V, D = np.linalg.eig(L)
V = np.dot(mean_removed, V.reshape(-1, 1))  # Reshape V to 2D array
V = V[:, :image_signature]

all_image_signature = np.zeros((rest_of_the_images.shape[1], image_signature), dtype=np.float32)
for i in range(rest_of_the_images.shape[1]):
    all_image_signature[i, :] = np.dot(mean_removed[:, i].T, V)

# Recognition
p = random_image - mean_value
s = np.dot(p.T, V)
z = np.linalg.norm(all_image_signature - s, axis=1, ord=2)
i = np.argmin(z)
accuracy = (1 - z[i] / np.linalg.norm(s, ord=2)) * 100

# Display results (you may need to adjust the display code based on your Python environment)
plt.subplot(121)
plt.imshow(random_image.reshape(112, 92), cmap='gray')
plt.title('Looking for this Face', fontweight='bold', fontsize=16, color='red')

plt.subplot(122)
plt.imshow(rest_of_the_images[:, i].reshape(112, 92), cmap='gray')
plt.title(f'Recognition Completed\nAccuracy: {accuracy:.2f}%', fontweight='bold', fontsize=16, color='red')

plt.show()
