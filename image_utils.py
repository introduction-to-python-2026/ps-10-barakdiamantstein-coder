
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
def load_image(path):
    img = Image.open(path)
    image_array = np.array(img)
    return image_array

def edge_detection(image):
    grey = np.mean(image, axis=2)
    kernelx = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])
    
    kernely = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    edgex = convolve2d(gray,kernelx, mode='same')
    edgey = convolve2d(gray,kernely, mode='same')
    edgeMAX = np.sqrt(edgex**2 + edgey**2)
    return edgeMAX
