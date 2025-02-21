import cv2 
import numpy as np

img = cv2.imread('images/DailyArt2.jpg')
gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# read template in gray
template_img = cv2.imread('images/part.jpg' , 0 )


res = cv2.matchTemplate(gray_img , template_img , cv2.TM_CCOEFF)
min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)


# creating bounding box

height , width = template_img.shape
top_left = max_loc
bottom_right = ( top_left[0] + width , top_left[1] + height )
cv2.rectangle(img , top_left , bottom_right , (0,0,255) , 5 )


cv2.imshow('hello' , img )
cv2.waitKey()
cv2.destroyAllWindows()