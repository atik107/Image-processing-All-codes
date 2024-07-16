import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


#blur filter to smooth image
def generate_gaussian_kernel(sigmaX, sigmaY, MUL=7, center_x=-1, center_y=-1):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1

    center_x = w // 2 if center_x == -1 else center_x
    center_y = h // 2 if center_y == -1 else center_y

    kernel = np.zeros((w, h))
    c = 1 / (2 * math.pi * sigmaX * sigmaY)

    for x in range(w):
        for y in range(h):
            dx = x - center_x
            dy = y - center_y

            x_part = (dx * dx) / (sigmaX * sigmaX)
            y_part = (dy * dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp(- 0.5 * (x_part + y_part))

    formatted_kernel = (kernel / np.min(kernel)).astype(int)

    return formatted_kernel

def sobel_derivatives():
    # Sobel x derivative kernel
    kernel_dx = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])

    # Sobel y derivative kernel
    kernel_dy = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])

    # Perform convolution for x derivative
    #derivative_x = convolve(image, kernel_dx)

    # Perform convolution for y derivative
    #derivative_y = convolve(image, kernel_dy)

    return kernel_dx, kernel_dy


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





#Image related Function
def normalize(image):
    cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)


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



def perform_convolution(imagePath, kernel,kernel_center=(-1, -1)):

    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    out_conv = convolve(image=image, kernel=kernel, kernel_center=kernel_center)
    #out_noramlize = normalize(out_conv)

    #cv2.imshow('Input image', image)
    #cv2.imshow('Covulated image', out_noramlize)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out_conv


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
    while (abs(dif - t) < 0.1 ** 6):
        t = dif
        dif = find_avg(image=image, t=t)

    return dif


def make_binary(t, image, low=0, high=255):
    out = image.copy()
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x, y]

            out[x, y] = high if v > t else low  # high--->white

    return out


def perform_threshold(image, lowThresholdRatio=0.09, highThresholdRatio=0.18):
    highThreshold = image.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(50)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)



def perform_hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == weak):
                if (
                    (image[i+1, j-1] == strong) or (image[i+1, j] == strong) or
                    (image[i+1, j+1] == strong) or (image[i, j-1] == strong) or
                    (image[i, j+1] == strong) or (image[i-1, j-1] == strong) or
                    (image[i-1, j] == strong) or (image[i-1, j+1] == strong)
                ):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out



def perform_edge_detection(image_path,sigma):
    #print("Image Path:", image_path)
    #kernel_dx, kernel_dy=sobel_derivatives();
    kernel_dx, kernel_dy = generate_gaussian_derivative(sigmaX=sig_x, sigmaY=sig_y, MUL=7, center_x=c_x, center_y=c_y)
    img1 = perform_convolution(imagePath=image_path, kernel=kernel_dx, kernel_center=(c_x, c_y))
    img2 = perform_convolution(imagePath=image_path, kernel=kernel_dy, kernel_center=(c_x, c_y))
    display_kernel(kernel_dx)
    display_kernel(kernel_dy)
    #print("img1 shape:", img1.shape)
    #print("img2 shape:", img2.shape)

    squared_img1 = np.square(img1)
    squared_img2 = np.square(img2)

    summ = squared_img1 + squared_img2
    merged_img = np.sqrt(summ)

    #theta = np.arctan2(conv_x, conv_y)
    theta = np.arctan2( img1, img2)
    merged_img_nor = normalize(merged_img)

    t = find_threeshold(image=merged_img_nor)
    #print(f"Threeshold {t}")
    final_out = make_binary(t=t, image=merged_img_nor, low=0, high=100)

    # Display all images
    images = [img1, img2, merged_img_nor, final_out]
    titles = ['X derivative', 'Y derivative', 'Merged', 'Thresholded']
    display_images(images, titles)

    # return final_out, theta
    return merged_img, theta



def perform_non_maximum_suppression(image, theta):
    image = image

    image = image / image.max() * 255 #normalize and scaling

    M, N = image.shape
    out = np.zeros((M, N), dtype=np.uint8)

    angle = theta * 180. / np.pi

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 0
            r = 0

            ang = angle[i, j]

            if (-22.5 <= ang < 22.5) or (157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]

            elif (-67.5 <= ang <= -22.5) or (112.5 <= ang <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]

            elif (67.5 <= ang <= 112.5) or (-112.5 <= ang <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]

            elif (22.5 <= ang < 67.5) or (-167.5 <= ang <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0
    return out

def plot_edge_directions(theta):
    # Create a quiver plot to visualize edge directions
    plt.figure(figsize=(8, 6))
    plt.imshow(theta, cmap='hsv', interpolation='nearest')
    plt.colorbar()
    plt.title('Edge Directions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def perform_canny(image_path, sigma):
    # Gray Scale Coversion
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    main_image = image

    # Perform Gaussian Blurr (Noise Reduction)
    kernel = generate_gaussian_kernel(sigmaX=sigma, sigmaY=sigma, MUL=7)
    image = convolve(image=image, kernel=kernel)

    cv2.imshow("Blurred Input Image", normalize(image))
    cv2.waitKey(0)

    # Gradient Calculation
    image_sobel, theta = perform_edge_detection(image_path,sigma)

    # Non Maximum Suppression
    suppressed = perform_non_maximum_suppression(image=image_sobel, theta=theta)

    # Threesholding and hysteresis
    threes, weak, strong = perform_threshold(image=suppressed)
    final_output = perform_hysteresis(image=threes, weak=weak, strong=strong)

    # Display all images
    plot_edge_directions(theta)
    images = [image, image_sobel, suppressed,final_output]
    titles = ['Original Image', 'Gradient Magnitude image', 'Non-maximum Suppression', 'Hysteresis thresholding']
    display_images(images, titles)



# Function to display the kernel as an image
def display_kernel(kernel):
    plt.imshow(kernel, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Kernel')
    plt.show()


def display_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()


# Outputs
kernel = None

sig_x = float(input("Enter Sigma(X): "))
#low_threshold_ratio = float(input("Enter Low Threshold Ratio: "))
#high_threshold_ratio = float(input("Enter High Threshold Ratio: "))


# sig_y = float(input("Enter Sigma(Y): "))
# c_x = int(input("Enter Center(X): "))
# c_y = int(input("Enter Center(Y): "))
sig_y = sig_x;
c_x = -1
c_y = -1


# Input raw images
image_path = '../images/lena.jpg'

perform_canny(image_path=image_path, sigma=sig_x)
#perform_canny(image_path=image_path, sigma=sig_x, l=low_threshold_ratio, h=high_threshold_ratio)