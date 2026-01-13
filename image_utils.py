
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(file_path):
    """
    Load a color image from a file path and return it as a NumPy array (uint8, 0-255)
    """
    from PIL import Image
    img = Image.open(file_path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def edge_detection(image_array):
    """
    Perform Sobel edge detection on a NumPy array image.
    Works with grayscale (2D) or RGB (3D) images.
    Works with float images (0-1) or uint8 images (0-255).
    
    Returns: 2D edge magnitude array (uint8, 0-255)
    """
    # Convert RGB to grayscale if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Convert float 0-1 to 0-255 uint8 if needed
    if image_array.dtype in [np.float32, np.float64]:
        image_array = (image_array * 255).astype(np.uint8)

    gray = image_array.astype(np.float64)

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

    # Normalize to 0-255
    edges = (edges / edges.max()) * 255
    edges = edges.astype(np.uint8)

    # Display the edges (optional)
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.show()

    return edges
