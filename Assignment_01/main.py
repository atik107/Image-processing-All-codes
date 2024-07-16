import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage as ndi

def segment_tree(image):
    # Load the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    # Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Adaptive Gaussian Thresholding
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    # Morphological opening and closing
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Detect contours and filter by area and circularity
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000 and is_circular(cnt)]

    # Draw contours on the original image
    result_image = image_rgb.copy()
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)

    return result_image, len(filtered_contours)

def is_circular(contour, circularity_threshold=0.75):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity > circularity_threshold

# Load the image
image_path = "../Tree_images/tree_seg_1.png"
original_image = cv2.imread(image_path)
segmented_image, num_rings = segment_tree(original_image)

# Show the result
plt.figure(figsize=(10, 8))
plt.title(f'Detected Rings: {num_rings}')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.show()
