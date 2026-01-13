
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
from PIL import Image

# 1️⃣ Load original color image
original_image = load_image('coes aerodynamics.jpg') 

# 2️⃣ Apply median filter for noise suppression
# Median expects a grayscale image or single channel; convert RGB to grayscale first
from skimage.color import rgb2gray
gray_image = rgb2gray(original_image)  # float image 0-1

# Apply median filter
clean_image = median(gray_image, ball(3))  # ball(3) = filter size, can change
clean_image = (clean_image * 255).astype('uint8')  # convert back to 0-255 uint8

# 3️⃣ Detect edges
edgeMAG = edge_detection(clean_image)

# 4️⃣ Save the edge-detected image
edge_image = Image.fromarray(edgeMAG)
edge_image.save("edge_image.png")

# 5️⃣ Optional: display original and edge-detected images side by side
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(edgeMAG, cmap="gray")
plt.title("Edge-Detected Image")
plt.axis("off")

plt.show()
