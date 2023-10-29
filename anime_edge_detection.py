import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


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

    # Visualize the results using matplotlib
    plt.figure(figsize=(15, 7))

    # Displaying the [0, 1] range version with 'gray' colormap
    plt.subplot(1, 2, 1)
    plt.imshow(clearer_edges, cmap='gray')
    plt.title("Clear Edges [0, 1] range")
    plt.axis('off')

    # Displaying the [0, 255] range version with 'jet' colormap
    plt.subplot(1, 2, 2)
    plt.imshow(clearer_edges_rescaled, cmap='jet')
    plt.title("Clear Edges [0, 255] range")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # cv2.imwrite('C:/Users/sundr/downloads/clearer_edges.jpg', clearer_edges_rescaled)


image_path = 'C:/Users/sundr/Downloads/fubuki.jpeg'  # Path of my image
load_and_detect_edges(image_path)
