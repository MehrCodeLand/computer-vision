import numpy as np 
import cv2 


img = cv2.imread('images/red.png')
img = cv2.resize(img ,(300,300) )
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
_ , binary_image = cv2.threshold(gray , 128 , 255 , cv2.THRESH_BINARY)
pixels = binary_image.flatten()


def rle_encoding(flat_image):

    values = []
    counts = []
    how_far_eq = 0 
    last = 0
    for i in range(0 , len(flat_image)):
        if flat_image[i] == flat_image[i+1]:
            how_far_eq += 1
            last = 0 
        else:
            counts.append(how_far_eq)
            values.append(flat_image[i - 1])
            how_far_eq = 0
            last = 1 
    
    if last == 1 : 
        counts.append(1)
        values.append(flat_image[-1])

    return values , counts





values , counts = rle_encoding(pixels)

print(values)

# cv2.imshow('color' , binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################
# tic tac toe simulator with ML 
# compression image

