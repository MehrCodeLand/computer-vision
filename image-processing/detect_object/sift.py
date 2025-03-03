import cv2
import numpy as np

img = cv2.imread('images/part.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use SIFT_create() to create a SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints
keypoints = sift.detect(gray, None)
print('key point : ', len(keypoints))

# Draw keypoints
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
