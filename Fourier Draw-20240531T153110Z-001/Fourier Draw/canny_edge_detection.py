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

def get_orientation_sector(dx, dy):
    angle = np.pi / 8

    A = np.array([[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]])

    B = np.array([[dx], [dy]])

    C = np.matmul(A, B)

    dx_p, dy_p = C[0, 0], C[1, 0]

    if dy_p < 0:
        dx_p, dy_p = -dx_p, -dy_p

    s_theta = -1

    if dx_p >= 0 and dx_p >= dy_p:
        s_theta = 0
    elif dx_p >= 0 and dx_p < dy_p:
        s_theta = 1
    elif dx_p < 0 and -dx_p < dy_p:
        s_theta = 2
    elif dx_p < 0 and -dx_p >= dy_p:
        s_theta = 3

    return s_theta

def is_local_max(E_mag, u, v, s_theta, t_low):
    m_c = E_mag[u, v]
    m_l, m_r = -1, -1

    if m_c < t_low:
        return False
    else:
        if s_theta == 0:
            m_l = E_mag[u-1, v]
        elif s_theta == 1:
            m_l = E_mag[u-1, v-1]
        elif s_theta == 2:
            m_l = E_mag[u, v-1]
        elif s_theta == 3:
            m_l = E_mag[u-1, v+1]

        if s_theta == 0:
            m_r = E_mag[u+1, v]
        elif s_theta == 1:
            m_r = E_mag[u+1, v+1]
        elif s_theta == 2:
            m_r = E_mag[u, v+1]
        elif s_theta == 3:
            m_r = E_mag[u+1, v-1]

        return m_l <= m_c and m_c >= m_r

def trace_and_threshold(E_nms, E_bin, u0, v0, t_low):
    E_bin[u0, v0] = 1

    u_l = max(u0-1, 0)
    u_r= min(u0+1, M-1)
    v_t = max(v0-1, 0)
    v_b = min(v0+1, N-1)

    for u in range(u_l, u_r+1):
        for v in range(v_t, v_b+1):
            if E_nms[u, v] >= t_low and E_bin[u, v] == 0:
                trace_and_threshold(E_nms, E_bin, u, v, t_low)

    return

E_mag = cv2.imread('mag_image.jpg', cv2.IMREAD_GRAYSCALE)
E_nms = np.zeros_like(E_mag)
E_bin = np.zeros_like(E_mag)

t_high = 20 / 100 * np.max(E_mag)
t_low = 5 / 100 * np.max(E_mag)

I_x = cv2.imread('x_image.jpg', cv2.IMREAD_GRAYSCALE)
I_y = cv2.imread('y_image.jpg', cv2.IMREAD_GRAYSCALE)

M, N = E_mag.shape[0], E_mag.shape[1]

cv2.imshow("Magnitude Image", E_mag)
cv2.waitKey(0)

for u in range(1, M-1):
    for v in range(1, N-1):
        dx, dy = I_x[u, v], I_y[u, v]
        s_theta = get_orientation_sector(dx, dy)
        if is_local_max(E_mag, u, v, s_theta, t_low):
            E_nms[u, v] = E_mag[u, v]

for u in range(1, M-1):
    for v in range(1, N-1):
        if E_nms[u, v] >= t_high and E_bin[u, v] == 0:
            trace_and_threshold(E_nms, E_bin, u, v, t_low)

cv2.normalize(src = E_nms, dst = E_nms, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
E_nms = np.round(E_nms).astype(dtype = np.uint8)

cv2.imshow("NMS Image", E_nms)
cv2.waitKey(0)

cv2.normalize(src = E_bin, dst = E_bin, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
E_bin = np.round(E_bin).astype(dtype = np.uint8)

cv2.imshow("Binary Image", E_bin)
cv2.waitKey(0)

cv2.imwrite("bin_image.jpg", E_bin)

cv2.destroyAllWindows()
