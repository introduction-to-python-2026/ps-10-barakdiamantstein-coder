from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import cv2
import matplotlib.pyplot as plt

def edge_detection(file_path):
    kernely = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    kernelx = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])

    gray = np.array(Image.open(file_path).convert("L"))

    gx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernely)

    edges = np.sqrt(gx**2 + gy**2)

    edges = (edges / edges.max()) * 255
    edges = edges.astype(np.uint8)

    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.show()

    return edges
