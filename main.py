from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from skimage.color import rgb2gray
from PIL import Image

# 1. Load original image
original_image = load_image("your_image.jpg")

# 2. Convert to grayscale
gray_image = rgb2gray(original_image)  # returns float 0-1

# 3. Apply median filter for noise suppression
clean_image = median(gray_image, ball(3))
clean_image = (clean_image * 255).astype('uint8')  # convert to 0-255

# 4. Detect edges
edgeMAG = edge_detection(clean_image)

# 5. Save edge image
edge_image = Image.fromarray(edgeMAG)
edge_image.save("edge_image.png")
