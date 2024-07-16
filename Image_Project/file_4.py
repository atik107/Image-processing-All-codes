import math
import cv2
import numpy as np
from skimage import measure, morphology
from scipy import ndimage as ndi
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

def segment_tree(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = binary_image > threshold_value
    filled_image = ndi.binary_fill_holes(binary_image)
    labeled_image, _ = ndi.label(filled_image)
    regions = measure.regionprops(labeled_image)
    tree_region = max(regions, key=lambda x: x.area)
    tree_mask = labeled_image == tree_region.label
    tree_only = np.where(tree_mask[:, :, None], image_rgb, 255)
    result_image = cv2.cvtColor(tree_only.astype('uint8'), cv2.COLOR_RGB2BGR)
    return result_image

def apply_local_histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    equalized_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return equalized_image

def apply_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def remove_noise(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.medianBlur(image, 5)
    image = cv2.bilateralFilter(image, 100, 2, 2)
    return image

def remove_noise_opening(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return dilated_image, eroded_image

def closing(img, kernel_size=3):
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    closed_image = cv2.erode(dilated_image, kernel, iterations=2)
    return closed_image

def zhang_suen_thinning(image):
    binary_image = (image // 255).astype(np.uint8)
    skeleton = morphology.thin(binary_image)
    thinned_image = (skeleton * 255).astype(np.uint8)
    return thinned_image

def remove_small_components(img, min_size=50):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img

def draw_filtered_contours(image, contours, min_contour_area=100):
    global filtered_contours
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
    segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rings_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
        (0, 128, 128), (192, 192, 192), (255, 165, 0), (255, 192, 203), (75, 0, 130),
        (255, 105, 180), (173, 216, 230), (0, 255, 127), (255, 69, 0), (250, 128, 114)
    ]
    for i, contour in enumerate(filtered_contours):
        color = colors[i % len(colors)]
        cv2.drawContours(rings_image, [contour], -1, color, 2)
    return rings_image

def find_extreme_points(binary_image):
    white_points = np.column_stack(np.where(binary_image == 255))
    top_most = white_points[white_points[:, 0].argmin()]
    bottom_most = white_points[white_points[:, 0].argmax()]
    left_most = white_points[white_points[:, 1].argmin()]
    right_most = white_points[white_points[:, 1].argmax()]
    return top_most, bottom_most, left_most, right_most

def count_transitions(binary_image, middle_vertical, middle_horizontal):
    vertical_transitions_list = []
    horizontal_transitions_list = []

    def transitions(line):
        return np.count_nonzero(line[:-1] != line[1:])

    for offset in [0, 10, -10]:
        col = middle_horizontal + offset
        if 0 <= col < binary_image.shape[1]:
            vertical_line = binary_image[:, col]
            vertical_transitions_list.append(transitions(vertical_line))

    for offset in [0, 10, -10]:
        row = middle_vertical + offset
        if 0 <= row < binary_image.shape[0]:
            horizontal_line = binary_image[row, :]
            horizontal_transitions_list.append(transitions(horizontal_line))

    avg_vertical_transitions = np.mean(vertical_transitions_list) / 4
    avg_horizontal_transitions = np.mean(horizontal_transitions_list) / 4

    return avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list

class TreeAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tree Age Detection")
        self.image_path = None
        self.images = []
        self.current_index = 0
        self.processed_images = []
        self.tree_age = 0

        # Set up the GUI layout
        self.setup_gui()

    def setup_gui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.load_button = tk.Button(self.frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.process_button = tk.Button(self.frame, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_button.grid(row=0, column=1, padx=5, pady=5)

        self.back_button = tk.Button(self.frame, text="Previous", command=self.show_previous_image, state=tk.DISABLED)
        self.back_button.grid(row=0, column=2, padx=5, pady=5)

        self.next_button = tk.Button(self.frame, text="Next", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.grid(row=0, column=3, padx=5, pady=5)

        self.tree_age_label = tk.Label(self.frame, text="Tree Age:")
        self.tree_age_label.grid(row=0, column=4, padx=5, pady=5)

        self.tree_age_text = tk.Text(self.frame, height=1, width=10)
        self.tree_age_text.grid(row=0, column=5, padx=5, pady=5)

        self.fig1 = Figure(figsize=(4, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.frame)
        self.canvas1.get_tk_widget().grid(row=1, column=0, columnspan=2)

        self.fig2 = Figure(figsize=(4, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.frame)
        self.canvas2.get_tk_widget().grid(row=1, column=2, columnspan=2)

        self.fig3 = Figure(figsize=(4, 4), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.frame)
        self.canvas3.get_tk_widget().grid(row=1, column=4, columnspan=2)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image, self.ax1)
            self.process_button.config(state=tk.NORMAL)

    def process_image(self):
        self.process_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)

        segmented_image = segment_tree(self.image_path)
        local_equalized_image = apply_local_histogram_equalization(segmented_image)

        segmented_image = remove_noise(segmented_image)
        for i in range(0, 2):
            local_equalized_image = apply_local_histogram_equalization(local_equalized_image)

        gamma_corrected_image = apply_gamma_correction(local_equalized_image, gamma=1.5)
        segmented_image_gray = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(segmented_image_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        dilated_image, eroded_image = remove_noise_opening(thresh)
        img_median = remove_noise(dilated_image)
        skeleton = zhang_suen_thinning(img_median)
        skeleton = closing(skeleton, 3)
        skeleton = remove_small_components(skeleton)

        contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rings_image = draw_filtered_contours(segmented_image, contours, 100)
        num_rings = len(filtered_contours)

        top_most, bottom_most, left_most, right_most = find_extreme_points(skeleton)
        middle_vertical = (top_most[0] + bottom_most[0]) // 2
        middle_horizontal = (left_most[1] + right_most[1]) // 2

        avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list = count_transitions(skeleton, middle_vertical, middle_horizontal)
        self.tree_age = math.ceil((avg_vertical_transitions + avg_horizontal_transitions) / 2)
        self.tree_age_text.delete(1.0, tk.END)
        self.tree_age_text.insert(tk.END, str(self.tree_age))

        self.images = [
            ("Original Image", self.original_image),
            ("Segmented Image", segmented_image),
            ("Local Histogram Equalized Image", local_equalized_image),
            ("Gamma Corrected Image", gamma_corrected_image),
            ("Blurred Image", blurred),
            ("Adaptive Thresholding", thresh),
            ("Eroded Image", eroded_image),
            ("Dilated Image", dilated_image),
            ("After Median Filtering", img_median),
            ("Skeletonization", skeleton),
            ("Skeleton with Analysis Lines", skeleton)
        ]
        self.show_image()

    def show_image(self):
        if self.images:
            title, img = self.images[self.current_index]
            self.ax2.clear()
            self.ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.ax2.set_title(title)
            self.canvas2.draw()

            if self.current_index == 0:
                self.back_button.config(state=tk.DISABLED)
            else:
                self.back_button.config(state=tk.NORMAL)
            if self.current_index == len(self.images) - 1:
                self.next_button.config(state=tk.DISABLED)
            else:
                self.next_button.config(state=tk.NORMAL)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def display_image(self, img, ax):
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("Original Image")
        self.canvas1.draw()

# Initialize the Tkinter main window
root = tk.Tk()
app = TreeAnalyzerApp(root)
root.mainloop()
