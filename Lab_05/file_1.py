import cv2
import numpy as np
import math
from tabulate import tabulate


def calculate_metrics(binary_image):
    # Find contours of the object
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Return None if no contours are found

    contour = max(contours, key=cv2.contourArea)  # Assume largest contour is the object

    # Calculate area
    area = cv2.contourArea(contour)

    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)

    # Erosion to highlight structure
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Border extraction
    border_image = cv2.subtract(binary_image, eroded_image)

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

    return area, perimeter, max_diameter, form_factor, roundness, compactness


def process_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image at {image_path}. Please check the path and file integrity.")
        return None
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Calculate metrics
    return calculate_metrics(binary_image)


# Image paths
train_image_path = '../images/p1.png'  # Update path for train image
test_image_path = '../images/t1.jpg'  # Update path for test image

# Process images
train_metrics = process_image(train_image_path)
test_metrics = process_image(test_image_path)

if train_metrics and test_metrics:
    # Display results
    headers = ['Metric', 'Train Image', 'Test Image']
    metrics_names = ['Area', 'Perimeter', 'Max Diameter', 'Form Factor', 'Roundness', 'Compactness']
    table_data = [[name, train_metrics[i], test_metrics[i]] for i, name in enumerate(metrics_names)]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
else:
    print("Error processing one or more images.")
