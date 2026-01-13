Load the original image

from image_utils import load_image
image = load_image("your_image.jpg")


Suppress noise using median filter

from skimage.filters import median
from skimage.morphology import ball
clean_image = median(image, ball(3))  # or another value


Detect edges

from image_utils import edge_detection
edgeMAG = edge_detection(clean_image)


Save the edge-detected image

from PIL import Image
edge_image = Image.fromarray(edgeMAG)
edge_image.save("edge_image.png")
