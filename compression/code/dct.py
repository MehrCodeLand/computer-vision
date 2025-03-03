import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = "images/face3.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Step 2: Resize the image to the smallest version
resized_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)  # Resize to 32x32

# Step 3: Normalize pixel values for DCT
normalized_image = np.float32(resized_image) / 255.0  # Scale pixel values to range [0, 1]

# Step 4: Apply DCT (Discrete Cosine Transform)
dct_image = cv2.dct(normalized_image)

# Step 5: Visualize the original and DCT-transformed image
plt.figure(figsize=(10, 5))

# Original resized image
plt.subplot(1, 2, 1)
plt.title("Resized Image (Grayscale)")
plt.imshow(resized_image, cmap="gray")
plt.axis("off")

# DCT-transformed image
plt.subplot(1, 2, 2)
plt.title("DCT Coefficients")
plt.imshow(np.log(abs(dct_image) + 1), cmap="gray")  # Log scale for better visualization
plt.axis("off")

plt.tight_layout()
plt.show()
