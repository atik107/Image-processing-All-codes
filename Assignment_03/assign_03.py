import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_double_gaussian_hist(height, mean1, std1, mean2, std2):
    x = np.arange(height)
    gaussian1 = (1 / (std1 * np.sqrt(2 * np.pi))) * np.exp(-((x - mean1) ** 2) / (2 * std1 ** 2))
    gaussian2 = (1 / (std2 * np.sqrt(2 * np.pi))) * np.exp(-((x - mean2) ** 2) / (2 * std2 ** 2))
    hist = gaussian1 + gaussian2
    hist /= np.sum(hist)
    return hist


def histogram_matching(input_image, target_hist):
    hist = cv2.calcHist([input_image], [0], None, [256], [0, 256])
    hist /= np.sum(hist)
    hist_cumsum = np.cumsum(hist)

    lut = np.zeros(256)
    for i in range(256):
        #print(np.cumsum(target_hist))
        lut[i] = np.argmin(np.abs(hist_cumsum[i] - np.cumsum(target_hist)))

    output_image = cv2.LUT(input_image, lut.astype(np.uint8))
    return output_image


# Parameters for the double Gaussian distribution
mean1 = 30
std1 = 8
mean2 = 165
std2 = 20

# Height of the histogram (usually 256 for grayscale images)
height = 256

# Read input grayscale image
image_path = '../images/histogram.jpg'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Generate the target histogram
target_hist = generate_double_gaussian_hist(height, mean1, std1, mean2, std2)

# Apply histogram matching
output_image = histogram_matching(input_image, target_hist)
output_hist = cv2.calcHist([output_image], [0], None, [256], [0, 256])

# Plotting
plt.figure(figsize=(16, 8))

# Input and output images
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Output Image')
plt.axis('off')
plt.show()

# Plot the target histogram
plt.figure(figsize=(8, 6))
plt.bar(np.arange(256), target_hist, color='red')
plt.title('Target Histogram (Double Gaussian)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Plot input image, target histogram, PDF, and CDF for input and output images, and output histogram in separate figures

# Plot input image and histograms
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.hist(input_image.ravel(), bins=256, color='blue', alpha=0.5, label='Input Histogram')
plt.title('Histogram')
plt.legend()

# Plot output image histogram
plt.subplot(2, 2, 2)
plt.plot(output_hist, color='green', label='Output Histogram')
plt.title('Histogram of Output Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(np.arange(256), target_hist, color='red', label='Target PDF')
plt.plot(np.arange(256), cv2.calcHist([input_image], [0], None, [256], [0, 256]) / np.prod(input_image.shape),
         color='blue', alpha=0.5, label='Input PDF')
plt.title('Probability Density Function (PDF)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(np.cumsum(target_hist), color='red', label='Target CDF')
plt.plot(np.cumsum(cv2.calcHist([input_image], [0], None, [256], [0, 256]) / np.prod(input_image.shape)),
         color='blue', alpha=0.5, label='Input CDF')
plt.title('Cumulative Distribution Function (CDF)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot PDF and CDF of the output image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(np.arange(256), output_hist, color='green', label='Output PDF')
plt.title('Probability Density Function (PDF) of Output Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(output_hist), color='green', label='Output CDF')
plt.title('Cumulative Distribution Function (CDF) of Output Image')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')
plt.legend()

plt.tight_layout()
plt.show()
