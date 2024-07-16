import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
from tqdm import tqdm 
import matplotlib.animation as animation 

def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

def findContours(image):
    contours = []
    visited = set()

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    def within_bounds(x, y):
        return 0 <= x < len(image) and 0 <= y < len(image[0])

    def dfs(x, y, contour):
        contour.append((x, y))
        visited.add((x, y))

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if within_bounds(nx, ny) and (nx, ny) not in visited and image[nx][ny] != 0:
                dfs(nx, ny, contour)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i, j) not in visited and image[i][j] != 0:
                contour = []
                dfs(i, j, contour)
                contours.append(contour)

    return contours

def extractXYPoints(contours):
    x_list = []
    y_list = []
    for contour in contours:
        x_points, y_points = zip(*contour)
        x_list.extend(x_points)
        y_list.extend(-np.array(y_points))
    return x_list, y_list

edged_image = cv2.imread("Nepal/bin_image.jpg", cv2.IMREAD_GRAYSCALE)

contours = findContours(edged_image)

x_list = []
y_list = []

for i in range(len(contours)):
  for xy in contours[i]:
    x_list.append(xy[0])
    y_list.append(xy[1])

x_list = x_list - np.mean(x_list)
y_list = y_list - np.mean(y_list)

theta = np.arctan2(y_list, x_list)

r = np.sqrt(np.square(x_list) + np.square(y_list))

sorted_indices = np.argsort(theta)
x_list = np.array(x_list)[sorted_indices]
y_list = np.array(y_list)[sorted_indices]


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)

xlim_data = plt.xlim() 
ylim_data = plt.ylim()

plt.show()

t_list = np.linspace(0, tau, len(x_list)) # now we can relate f(t) -> x,y

# Now find forier coefficient from -n to n circles
# ..., c-3, c-2, c-1, c0, c1, c2, c3, ...
order = 300 # -order to order i.e -100 to 100
# you can change the order to get proper figure
# too much is also not good, and too less will not produce good result

print("generating coefficients ...")
# lets compute fourier coefficients from -order to order
c = []
pbar = tqdm(total=(order*2+1))
# we need to calculate the coefficients from -order to order
for n in range(-order, order+1):
    # calculate definite integration from 0 to 2*PI
    # formula is given in readme
    coef = 1/tau*quad_vec(lambda t: f(t, t_list, x_list, y_list)*np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
    c.append(coef)
    pbar.update(1)
pbar.close()
print("completed generating coefficients.")

# converting list into numpy array
c = np.array(c)

# save the coefficients for later use
np.save("coeff.npy", c)

## -- now to make animation with epicycle -- ##

# this is to store the points of last circle of epicycle which draws the required figure
draw_x, draw_y = [], []

# make figure for animation
fig, ax = plt.subplots()

# different plots to make epicycle
# there are -order to order numbers of circles
circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order+1)]
# circle_lines are radius of each circles
circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order+1)]
# drawing is plot of final drawing
drawing, = ax.plot([], [], 'k-', linewidth=2)

# original drawing
# orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# to fix the size of figure so that the figure does not get cropped/trimmed
ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)

# hide axes
ax.set_axis_off()

# to have symmetric axes
ax.set_aspect('equal')

# Set up formatting for the video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Amrit Aryal'), bitrate=1800)

print("compiling animation ...")
# set number of frames
frames = 300
pbar = tqdm(total=frames)

# save the coefficients in order 0, 1, -1, 2, -2, ...
# it is necessary to make epicycles
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order+1):
        new_coeffs.extend([coeffs[order+i],coeffs[order-i]])
    return np.array(new_coeffs)

# make frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    global pbar
    # get t from time
    t = time[i]

    # exponential term to be multiplied with coefficient 
    # this is responsible for making rotation of circle
    exp_term = np.array([np.exp(n*t*1j) for n in range(-order, order+1)])

    # sort the terms of fourier expression
    coeffs = sort_coeff(coeffs*exp_term) # coeffs*exp_term makes the circle rotate. 
    # coeffs itself gives only direction and size of circle

    # split into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    # center points for fisrt circle
    center_x, center_y = 0, 0

    # make all circles i.e epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # calculate radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff]) # similar to magnitude: sqrt(x^2+y^2)

        # draw circle with given radius at given center points of circle
        # circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        theta = np.linspace(0, tau, num=50) # theta should go from 0 to 2*PI to get all points of circle
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff
    
    # center points now are points from last circle
    # these points are used as drawing points
    draw_x.append(center_x)
    draw_y.append(center_y)

    # draw the curve from last point
    drawing.set_data(draw_x, draw_y)

    # draw the real curve
    # orig_drawing.set_data(x_list, y_list)

    # update progress bar
    pbar.update(1)

# make animation
# time is array from 0 to tau 
time = np.linspace(0, tau, num=frames)
anim = animation.FuncAnimation(fig, make_frame, frames=frames, fargs=(time, c),interval=5)
dpi = 300
anim.save('epicycle.mp4', writer=writer, dpi=dpi)
pbar.close()
print("completed: epicycle.mp4")