import cv2
import numpy as np


img = cv2.imread('images/part.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()

# keypoint
keypoints = sift.detect(gray, None)
print( 'key point : ', len(keypoints))


img = cv2.drawKeypoints(img , keypoints , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()