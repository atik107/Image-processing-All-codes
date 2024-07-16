import math
import cv2
import numpy as np
from skimage import measure, morphology
from scipy import ndimage as ndi
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TreeAgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tree Age Detection")

        self.image_path = None
        self.processed_images = []
        self.current_image_index = 0

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack()

        self.load_button = tk.Button(frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.process_button = tk.Button(frame, text="Process Image", command=self.process_image)
        self.process_button.grid(row=0, column=1, padx=5, pady=5)

        self.prev_button = tk.Button(frame, text="Previous", command=self.prev_image)
        self.prev_button.grid(row=0, column=2, padx=5, pady=5)

        self.next_button = tk.Button(frame, text="Next", command=self.next_image)
        self.next_button.grid(row=0, column=3, padx=5, pady=5)

        self.age_label = tk.Label(frame, text="Tree Age: ")
        self.age_label.grid(row=0, column=4, padx=5, pady=5)

        self.fig_input = Figure(figsize=(4, 4))
        self.ax_input = self.fig_input.add_subplot(111)
        self.canvas_input = FigureCanvasTkAgg(self.fig_input, master=self.root)
        self.canvas_input.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_steps = Figure(figsize=(4, 4))
        self.ax_steps = self.fig_steps.add_subplot(111)
        self.canvas_steps = FigureCanvasTkAgg(self.fig_steps, master=self.root)
        self.canvas_steps.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_output = Figure(figsize=(4, 4))
        self.ax_output = self.fig_output.add_subplot(111)
        self.canvas_output = FigureCanvasTkAgg(self.fig_output, master=self.root)
        self.canvas_output.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ax_input.clear()
            self.ax_input.imshow(image_rgb)
            self.canvas_input.draw()

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded")
            return

        image = cv2.imread(self.image_path)
        processed_steps, tree_age = self.segment_and_process_tree(image)

        self.processed_images = processed_steps
        self.current_image_index = 0

        self.display_images()
        self.age_label.config(text=f"Tree Age: {tree_age} years")

    def next_image(self):
        if self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.display_images()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_images()

    def display_images(self):
        input_image = self.processed_images[0]
        current_step_image = self.processed_images[self.current_image_index]
        final_image = self.processed_images[-1]

        self.ax_input.clear()
        self.ax_input.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        self.canvas_input.draw()

        self.ax_steps.clear()
        self.ax_steps.imshow(cv2.cvtColor(current_step_image, cv2.COLOR_BGR2RGB))
        self.canvas_steps.draw()

        self.ax_output.clear()
        self.ax_output.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        self.canvas_output.draw()

    def segment_and_process_tree(self, image):
        processed_steps = []

        # Segment the tree
        segmented_image = self.segment_tree(image)
        processed_steps.append(segmented_image)

        # Apply local histogram equalization
        local_equalized_image = self.apply_local_histogram_equalization(segmented_image)
        processed_steps.append(local_equalized_image)

        # Apply gamma correction
        gamma_corrected_image = self.apply_gamma_correction(local_equalized_image, gamma=1.5)
        processed_steps.append(gamma_corrected_image)

        # Convert to grayscale
        gray_image = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        processed_steps.append(thresh)

        # Morphological operations
        dilated_image, eroded_image = self.remove_noise_opening(thresh)
        processed_steps.append(eroded_image)
        processed_steps.append(dilated_image)

        img_median = self.remove_noise(dilated_image)
        processed_steps.append(img_median)

        # Thinning
        skeleton = self.zhang_suen_thinning(img_median)
        skeleton = self.closing(skeleton, 3)
        skeleton = self.remove_small_components(skeleton)
        processed_steps.append(skeleton)

        # Draw filtered contours
        contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rings_image = self.draw_filtered_contours(segmented_image, contours, 100)
        processed_steps.append(rings_image)

        # Count transitions
        top_most, bottom_most, left_most, right_most = self.find_extreme_points(skeleton)
        middle_vertical = (top_most[0] + bottom_most[0]) // 2
        middle_horizontal = (left_most[1] + right_most[1]) // 2
        avg_vertical_transitions, avg_horizontal_transitions, _, _ = self.count_transitions(skeleton, middle_vertical, middle_horizontal)

        age = math.ceil((avg_vertical_transitions + avg_horizontal_transitions) / 2)

        return processed_steps, age

    def segment_tree(self, image):
        # Segment the tree from the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = binary_image > 0
        filled_image = ndi.binary_fill_holes(binary_image)
        labeled_image, _ = ndi.label(filled_image)
        regions = measure.regionprops(labeled_image)
        tree_region = max(regions, key=lambda x: x.area)
        tree_mask = labeled_image == tree_region.label
        tree_only = np.where(tree_mask[:, :, None], image_rgb, 255)
        return cv2.cvtColor(tree_only.astype('uint8'), cv2.COLOR_RGB2BGR)

    def apply_local_histogram_equalization(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_gamma_correction(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def remove_noise(self, image):
        image = cv2.medianBlur(image, 3)
        image = cv2.medianBlur(image, 5)
        return cv2.bilateralFilter(image, 100, 2, 2)

    def remove_noise_opening(self, image):
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        return dilated_image, eroded_image

    def closing(self, img, kernel_size=3):
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
        return cv2.erode(dilated_image, kernel, iterations=2)

    def zhang_suen_thinning(self, image):
        binary_image = (image // 255).astype(np.uint8)
        skeleton = morphology.thin(binary_image)
        return (skeleton * 255).astype(np.uint8)

    def remove_small_components(self, img, min_size=50):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                img[labels == i] = 0
        return img

    def draw_filtered_contours(self, image, contours, min_contour_area=100):
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
        segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rings_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (255, 165, 0), (255, 192, 203), (75, 0, 130), (255, 105, 180), (173, 216, 230), (0, 255, 127),
            (255, 69, 0), (250, 128, 114)
        ]
        for i, contour in enumerate(filtered_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(rings_image, [contour], -1, color, 2)
        return rings_image

    def find_extreme_points(self, binary_image):
        white_points = np.column_stack(np.where(binary_image == 255))
        top_most = white_points[white_points[:, 0].argmin()]
        bottom_most = white_points[white_points[:, 0].argmax()]
        left_most = white_points[white_points[:, 1].argmin()]
        right_most = white_points[white_points[:, 1].argmax()]
        return top_most, bottom_most, left_most, right_most

    def count_transitions(self, binary_image, middle_vertical, middle_horizontal):
        def transitions(line):
            return np.count_nonzero(line[:-1] != line[1:])
        vertical_transitions_list = [transitions(binary_image[:, col]) for col in [middle_horizontal, middle_horizontal + 10, middle_horizontal - 10] if 0 <= col < binary_image.shape[1]]
        horizontal_transitions_list = [transitions(binary_image[row, :]) for row in [middle_vertical, middle_vertical + 10, middle_vertical - 10] if 0 <= row < binary_image.shape[0]]
        avg_vertical_transitions = np.mean(vertical_transitions_list) / 4
        avg_horizontal_transitions = np.mean(horizontal_transitions_list) / 4
        return avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list

if __name__ == "__main__":
    root = tk.Tk()
    app = TreeAgeApp(root)
    root.mainloop()
