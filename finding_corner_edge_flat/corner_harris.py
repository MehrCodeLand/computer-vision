import cv2
import numpy as np


img = cv2.imread('images/chess.jpg')
gray =  cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
harris_corner = cv2.cornerHarris(gray , 3 , 3 , 0.05 )


kernel = np.ones((7,7) , np.uint8)
harris_corner = cv2.dilate(harris_corner , kernel , iterations=2)

img[harris_corner > 0.025 * harris_corner.max() ] = [255 , 127 , 127]


cv2.imshow('hello' , img)
cv2.waitKey()
cv2.destroyAllWindows()