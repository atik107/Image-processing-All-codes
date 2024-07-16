import numpy as np
import cv2

def gen_gaussian(x, y, sigma):
    return 1 / (2 * np.pi * sigma**2) * np.exp(- (x**2 + y**2) / (2 * sigma**2))

def gen_derivatives(M, N, sigma):
    left = M // 2
    top = N // 2
    
    x_derivative = np.zeros((M, N))
    y_derivative = np.zeros((M, N))
    
    for i in range(-left, left+1):
        for j in range(-top, top+1):
            x_derivative[i+left, j+top] = - i / sigma**2 * gen_gaussian(i, j, sigma)
            y_derivative[i+left, j+top] = - j / sigma**2 * gen_gaussian(i, j, sigma)
    
    return x_derivative, y_derivative

def applyFilter(img, kernel, cx = None, cy = None):
    
    if cx == None:
        cx = kernel.shape[0] // 2
    if cy == None:
        cy = kernel.shape[1] // 2
            
    kleft = -(cx - 0)
    kright = (kernel.shape[0] - 1) - cx 
    ktop = -(cy - 0)
    kbottom = (kernel.shape[1] - 1) - cy
    
    bordered_img = cv2.copyMakeBorder(src = img, top = -ktop, bottom = kbottom, left = -kleft, right = kright, borderType = cv2.BORDER_CONSTANT)
    out = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for x in range(ktop, kbottom + 1):
                for y in range(kleft, kright + 1):
                    out[i, j] += bordered_img[(i - ktop) + x, (j - kleft) + y] * kernel[(kernel.shape[0] - 1) - (x - ktop), (kernel.shape[1] - 1) - (y - kleft)]
    
    return out

def apply_grad_mag(x_image, y_image):
    M = x_image.shape[0]
    N = y_image.shape[1]
    
    output = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            output[i, j] = np.sqrt(x_image[i, j]**2 + y_image[i, j]**2)
    
    return output


img = cv2.imread('bangladesh.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale", img)
#cv2.waitKey(0)

x_kernel, y_kernel = gen_derivatives(7, 7, 1)

x_image = applyFilter(img, x_kernel)
cv2.imwrite("x_image.jpg", x_image)
#cv2.normalize(src = x_image, dst = x_image, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
#x_image = np.round(x_image).astype(dtype = np.uint8)

#cv2.imshow("X derivative", x_image)
#cv2.waitKey(0)

y_image = applyFilter(img, y_kernel)
cv2.imwrite("y_image.jpg", y_image)
#cv2.normalize(src = y_image, dst = y_image, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
#y_image = np.round(y_image).astype(dtype = np.uint8)

#cv2.imshow("Y derivative", y_image)
#cv2.waitKey(0)

mag_image = apply_grad_mag(x_image, y_image)
cv2.normalize(src = mag_image, dst = mag_image, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
mag_image = np.round(mag_image).astype(dtype = np.uint8)

cv2.imwrite("mag_image.jpg", mag_image)
cv2.imshow("Magnitude Image", mag_image)
cv2.waitKey(0)



"""
def apply_threshold(image, delta):
    T = np.average(image)
    M, N = image.shape[0], image.shape[1]
    temp = np.zeros((M, N))
    
    while True:
        
        g1 = []
        g2 = []
        
        for i in range(M):
            for j in range(N):
                if image[i, j] > T:
                    g1.append(image[i, j])
                else:
                    g2.append(image[i, j])
                    
        g1 = np.array(g1)
        g2 = np.array(g2)
        
        avg1 = np.average(g1)
        avg2 = np.average(g2)
        
        Tt = (avg1 + avg2) / 2
        
        if abs(Tt - T) < delta:
            break
        
        T = Tt
    
    for i in range(M):
        for j in range(N):
            if image[i, j] > T:
                temp[i, j] = 1
            else:
                temp[i, j] = 0
                
    return temp

mag_img = cv2.imread("magimage.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Magnitude Image", mag_img)
cv2.waitKey(0)

temp = apply_threshold(mag_img, 1e-9)
cv2.imshow("Threshold Image", temp)
cv2.waitKey(0)
"""
cv2.destroyAllWindows()