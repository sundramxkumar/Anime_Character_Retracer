    # Words from the dev : There's still something lacking in this piece of code.
    # I tried three types different logic to draw this.
    # But the live_draw() is keeps printing the white line instead of the actual image processed.
    # It will be under maintenance for now.
    # If you could resolve this issue.
    # Please commit ur changes and ping me to commit ur version in the master/main file
    # Thanks, im still a beginner, so it would mean a lot to me.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from matplotlib.collections import LineCollection


def load_and_detect_edges(path):
    # Load the image in color mode
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection. Adjust the thresholds for desired clarity.
    edges = cv2.Canny(blurred_image, 50, 150)

    # Dilate the edges to make them thicker
    dilated_edges = cv2.dilate(edges, None, iterations=1)

    # Erode the edges to refine them
    eroded_edges = cv2.erode(dilated_edges, None, iterations=1)

    # Enhance the contrast and clarity of the resulting edge image
    clearer_edges = exposure.equalize_adapthist(eroded_edges / np.max(np.array(eroded_edges)), clip_limit=0.03)
    clearer_edges_rescaled = (clearer_edges * 255).astype(np.uint8)

    # Move directly to the live drawing simulation
    live_draw(clearer_edges_rescaled)


def live_draw(image, delay=0.001, skip=5):  # New skip parameter
    # Get coordinates of pixels with intensity values above a threshold
    threshold = 240
    y, x = np.where(image >= threshold)

    points = list(zip(x, y))
    # Sort the points for a more structured drawing.
    points.sort(key=lambda p: (p[1], p[0]))

    # Skip pixels for faster drawing
    points = points[::skip]

    # Create line segments from consecutive points
    lines = [points[i:i + 2] for i in range(0, len(points), 2)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.zeros_like(image), cmap='gray')
    ax.axis('off')

    line_segments = LineCollection([], colors='white', linewidths=1.0)  # Increased line width
    ax.add_collection(line_segments)

    for i in range(len(lines)):
        line_segments.set_segments(lines[:i])
        plt.draw()
        plt.pause(delay)

    plt.show()


# Test the function
image_path = 'C:/Users/sundr/PycharmProjects/AnimeEdgeDetection/fubuki.jpeg'
load_and_detect_edges(image_path)
