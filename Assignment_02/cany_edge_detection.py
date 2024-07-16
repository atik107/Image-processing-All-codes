import cv2
import matplotlib.pyplot as plt


def apply_canny_edge_detection(image, sigma, low_threshold, high_threshold):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges


# Load an image
image_path = '../images/lena.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Set parameters
sigma = 5
low_threshold = 80
high_threshold = 100

# Apply Canny edge detection
edges = apply_canny_edge_detection(image, sigma, low_threshold, high_threshold)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Edges Detected')

plt.show()
