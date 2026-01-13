import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def load_image(file_path):
    """
    Load a color image from a file path and return it as a NumPy array.
    """
    img = Image.open(file_path).convert("RGB")
    return np.array(img)

def edge_detection(image_array):
    """
    Perform Sobel edge detection on a NumPy array image (grayscale or RGB).
    Returns the edge magnitude array.
    """
    # Convert to grayscale if the image is RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    gray = gray.astype(np.float64)

    # Sobel kernels
    kernelx = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])
    
    kernely = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    # Apply Sobel filters
    gx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernely)

    # Compute edge magnitude
    edges = np.sqrt(gx**2 + gy**2)
    edges = (edges / edges.max()) * 255
    edges = edges.astype(np.uint8)

    # Display edges (optional)
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.show()

    return edges
