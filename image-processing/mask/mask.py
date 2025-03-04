import cv2
import numpy as np

# Load image
img = cv2.imread("images/face.png")
# else 
img2 = cv2.imread('images/face2.png') 
img2 = cv2.resize(img2 , (img.shape[1] , img.shape[0] ))

# Create a blank mask (same size as the image)
mask = np.zeros(img.shape[:2], dtype="uint8")

# Define a circular region for masking
cv2.circle(mask, (200, 200), 100, 255, -1)

# Apply the mask
result = cv2.bitwise_and(img, img, mask=mask)

# Show the images
cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Masked Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
