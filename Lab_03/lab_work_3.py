import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
image_path = '../images/histogram.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



#input image
histogram = cv2.calcHist([image],[0], None, [256], [0,256])
pdf_input = histogram / np.sum(histogram)
cdf_input = np.cumsum(pdf_input)

#histogram equalization
equalized_image = np.zeros_like(image)
cdf_min = np.min(cdf_input)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        tmp=image[i, j]
        t=(cdf_input[tmp]) * 255
        equalized_image[i, j] = np.round(t)

#equalized image
equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
pdf_equalized = equalized_histogram / np.sum(equalized_histogram)
cdf_equalized = np.cumsum(pdf_equalized)





#img+histogram

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



#pdf+cdf

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

#plt.tight_layout()
plt.show()