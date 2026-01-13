from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import matplotlib.pyplot as plt

# 1️⃣ Load the original color image
original_image = load_image("cow aerodynamics.jpg") 

# 2️⃣ Apply noise suppression using median filter
clean_image = median(original_image, ball(3))

# 3️⃣ Detect edges on the noise-free image
edgeMAG = edge_detection(clean_image)

# 4️⃣ Convert edge magnitude to PIL Image and save
edge_image = Image.fromarray(edgeMAG)
edge_image.save("edge_image.png")

# 5️⃣ Optional: display original and edge images for verification
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edgeMAG, cmap="gray")
plt.title("Edge-Detected Image")
plt.axis("off")

plt.show()
