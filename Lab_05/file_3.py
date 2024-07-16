import cv2
import numpy as np
import math
from tabulate import tabulate

def calculate_metrics(binary_image):
    # Find contours of the object
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Handle case where no contours are found

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Erosion to highlight structure
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Calculate Max Diameter
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    max_diameter = max(np.linalg.norm(np.array(leftmost) - np.array(rightmost)),
                       np.linalg.norm(np.array(topmost) - np.array(bottommost)))

    # Calculate Form Factor, Roundness, and Compactness
    form_factor = (4 * math.pi * area) / (perimeter ** 2)
    roundness = (4 * area) / (math.pi * max_diameter ** 2)
    compactness = perimeter / math.sqrt(area)

    return [area, perimeter, max_diameter, form_factor, roundness, compactness]

def process_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image at {image_path}.")
        return None
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Calculate metrics
    return calculate_metrics(binary_image)

def euclidean_distance(features1, features2):
    # Calculate Euclidean distance between two feature vectors
    return np.sqrt(np.sum((np.array(features1) - np.array(features2))**2))

# Define paths to your images
img1 = '../images/p1.png'
img2 = '../images/p2.png'
img3 = '../images/p3.jpg'

t1 = '../images/t1.jpg'
t2 = '../images/t2.jpg'
t3 = '../images/st.jpg'
#t4 = '../images/t1.jgp'

train_images = [img1, img2, img3]
test_images = [t1,t2,t3]

# Calculate metrics for all images
train_features = [process_image(img) for img in train_images]
test_features = [process_image(img) for img in test_images]

# Compute similarity matrix
similarity_matrix = []
for test in test_features:
    if test:
        similarities = [euclidean_distance(test, train) if train else float('inf') for train in train_features]
        similarity_matrix.append(similarities)
    else:
        similarity_matrix.append([float('inf')] * len(train_images))

# Table headers
row_headers = [f'Test {i + 1}' for i in range(len(test_images))]
col_headers = [f'GT {i + 1}' for i in range(len(train_images))]

# Print table
print(tabulate(similarity_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))
