import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = '../images/histogram.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the grayscale image
cv2.imshow('Grayscale Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#New code

#histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

#equalized img
equalized_image = cv2.equalizeHist(image)

#histogram of equalized image
equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])



#Output show
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(histogram, color='black')
plt.title("Histogram of Input Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
plt.imshow(image, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(equalized_histogram, color='black')
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

plt.tight_layout()
plt.show()

#CDF and PDF of input and equalized images
cdf_input = np.cumsum(histogram)
pdf_input = cdf_input / np.sum(histogram)

cdf_equalized = np.cumsum(equalized_histogram)
pdf_equalized = cdf_equalized / np.sum(equalized_histogram)








#CDF and PDF of input and equalized images
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(cdf_input, color='black')
plt.title("CDF of Input Image")
plt.xlabel("Pixel Value")
plt.ylabel("Cumulative Probability")

plt.subplot(2, 2, 2)
plt.plot(pdf_input, color='black')
plt.title("PDF of Input Image")
plt.xlabel("Pixel Value")
plt.ylabel("Probability Density")

plt.subplot(2, 2, 3)
plt.plot(cdf_equalized, color='black')
plt.title("CDF of Equalized Image")
plt.xlabel("Pixel Value")
plt.ylabel("Cumulative Probability")

plt.subplot(2, 2, 4)
plt.plot(pdf_equalized, color='black')
plt.title("PDF of Equalized Image")
plt.xlabel("Pixel Value")
plt.ylabel("Probability Density")

plt.tight_layout()
plt.show()


