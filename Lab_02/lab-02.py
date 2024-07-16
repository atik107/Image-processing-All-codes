import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_derivative(sigmaX, sigmaY, MUL=7, center_x=-1, center_y=-1):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1

    center_x = w // 2 if center_x == -1 else center_x
    center_y = h // 2 if center_y == -1 else center_y

    kernel_dx = np.zeros((w, h))
    kernel_dy = np.zeros((w, h))
    c = 1 / (2 * math.pi * sigmaX * sigmaY)

    for x in range(w):
        for y in range(h):
            dx = x - center_x
            dy = y - center_y

            x_part = (dx * dx) / (sigmaX * sigmaX)
            y_part = (dy * dy) / (sigmaY * sigmaY)

            kernel_dx[x][y] = -c * (dx / (sigmaX * sigmaX)) * math.exp(-0.5 * (x_part + y_part))
            kernel_dy[x][y] = -c * (dy / (sigmaY * sigmaY)) * math.exp(-0.5 * (x_part + y_part))

    return kernel_dx, kernel_dy



def x_derivative(image, sigma):
    smoothed = generate_gaussian_derivative(image,sigmaX=sigma,sigmaY=sigma)
    derivative_x = np.gradient(smoothed, axis=1)
    return derivative_x

def y_derivative(image, sigma):
    smoothed = generate_gaussian_derivative(image, sigmaX=sigma,sigmaY=sigma)
    derivative_y = np.gradient(smoothed, axis=0)
    return derivative_y

# Function to display the kernel as an image
def display_kernel(kernel):
    plt.imshow(kernel, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Kernel')
    plt.show()



def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1

    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_image


def convolve(image, kernel, kernel_center=(-1, -1)):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    padded_image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype='float32')

    for y in range(kernel_height // 2, image_height + kernel_height // 2):
        for x in range(kernel_width // 2, image_width + kernel_width // 2):
            sum = 0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    sum += kernel[ky, kx] * padded_image[y - kernel_height // 2 + ky, x - kernel_width // 2 + kx]
            output[y - kernel_height // 2, x - kernel_width // 2] = sum

    return output


def normalize(image):
    cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)


def perform_convolution(imagePath, kernel,kernel_center=(-1, -1)):

    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    out_conv = convolve(image=image, kernel=kernel, kernel_center=kernel_center)
    #out_noramlize = normalize(out_conv)

    #cv2.imshow('Input image', image)
    #cv2.imshow('Covulated image', out_noramlize)
    return out_conv
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def find_avg(image, t=-1):
    total1 = 0
    total2 = 0
    c1 = 0
    c2 = 0

    h, w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x][y]
            if px > t:
                total2 += px
                c2 += 1
            else:
                total1 += px
                c1 += 1
    mu1 = total1 / c1
    mu2 = total2 / c2

    return (mu1 + mu2) / 2

def find_threeshold(image):
    total = 0
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x, y]
            total += px
    t = total / (h * w)

    dif = find_avg(image=image, t=t)
    while (abs(dif - t) < 0.1 ** 4):
        t = dif
        dif = find_avg(image=image, t=t)

    return dif


def make_binary(t, image, low=0, high=255):
    out = image.copy()
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x, y]

            out[x, y] = high if v > t else low

    return out
def plot_historgram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


print("Gaussian Kernel:")
sig_x = float(input("Enter Sigma(X): "))
#sig_y = float(input("Enter Sigma(Y): "))
#c_x = int(input("Enter Center(X): "))
#c_y = int(input("Enter Center(Y): "))
sig_y=sig_x;
c_x=-1
c_y=-1
kernel_dx, kernel_dy = generate_gaussian_derivative(sigmaX=sig_x, sigmaY=sig_y, MUL=7, center_x=c_x, center_y=c_y)

print("Kernel for dx:")
print(kernel_dx)
display_kernel(kernel_dx)

print("\nKernel for dy:")
print(kernel_dy)
display_kernel(kernel_dy)

# Input raw images
image_path = '../images/lena.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the grayscale image
cv2.imshow('Grayscale Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


img1 = perform_convolution(imagePath=image_path, kernel=kernel_dx,kernel_center=(c_x, c_y))
img2 = perform_convolution(imagePath=image_path, kernel=kernel_dy,kernel_center=(c_x, c_y))

squared_img1 = np.square(img1)
squared_img2 = np.square(img2)

summ = squared_img1 + squared_img2
magnitude_gradient = np.sqrt(summ)


#normalize
img1=normalize(img1)
img2=normalize(img2)
output_nor = normalize(magnitude_gradient)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot the x derivative
axes[0, 0].imshow(img1, cmap='gray')
axes[0, 0].set_title('x derivative')

# Plot the y derivative
axes[0, 1].imshow(img2, cmap='gray')
axes[0, 1].set_title('y derivative')

# Plot the magnitude of gradient
axes[1, 0].imshow(output_nor, cmap='gray')
axes[1, 0].set_title('Magnitude of Gradient')

# Plot histogram
axes[1, 1].hist(output_nor.ravel(), bins=256, color='black')
axes[1, 1].set_title('Histogram')

# Find threshold and create binary image
threshold = find_threeshold(image=output_nor)
print(f"Threshold: {threshold}")
final_out = make_binary(t=threshold, image=output_nor, low=0, high=255)
axes[1, 1].axvline(x=threshold, color='r', linestyle='--')
axes[1, 1].text(threshold, 100, f'Threshold: {threshold}', color='r')
axes[1, 1].set_xlim(0, 255)  # Set the x-axis limit to the range of pixel values

# Show thresholded image
axes[1, 1].imshow(final_out, cmap='gray', vmin=0, vmax=250)  # Ensure correct pixel value range

# Show the plot
plt.show()
