import cv2
import numpy as np
import math


def cosine_similarity(vectorA, vectorB):
    dot_product = sum(a * b for a, b in zip(vectorA, vectorB))
    magnitudeA = math.sqrt(sum(a * a for a in vectorA))
    magnitudeB = math.sqrt(sum(b * b for b in vectorB))  # Fixing the magnitude calculation

    if magnitudeA == 0 or magnitudeB == 0:
        return 0.0

    cosine_sim = dot_product / (magnitudeA * magnitudeB)
    return cosine_sim


def find_bounding_box(image):
    nonzero_pixels = np.nonzero(image)

    x1 = np.min(nonzero_pixels[1])
    y1 = np.min(nonzero_pixels[0])
    x2 = np.max(nonzero_pixels[1])
    y2 = np.max(nonzero_pixels[0])

    return (x1, y1, x2, y2)


def calculate_max_diameter(bbox):
    xmin, ymin, xmax, ymax = bbox
    max_diameter = max(xmax - xmin, ymax - ymin)
    return max_diameter


image_path = 'path_to_your_image/t1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found or the path is incorrect")

area = np.count_nonzero(image)

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)

eroded_image = cv2.erode(image, kernel, iterations=1)

subtracted_image = cv2.subtract(image, eroded_image)

perimeter = np.count_nonzero(subtracted_image)

bbox = find_bounding_box(image)
max_diameter = calculate_max_diameter(bbox)

vectorA = [1000, 50, 3]
vectorB = [area, perimeter, max_diameter]

print("Cosine Similarity:", cosine_similarity(vectorA, vectorB))
print("Area:", area, "Perimeter:", perimeter, "Max Diameter:", max_diameter)

cv2.imshow('Eroded Image', subtracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
