import cv2
import numpy as np


def skin_detection(image_path):

    img = cv2.imread('images/face5.png')
    img_YCrCb = cv2.cvtColor(img , cv2.COLOR_BGR2YCrCb)
    img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

    lower_ycrcb , upper_ycrcb = find_range_YCrCb(img_YCrCb)
    skin_mask_ycrcb = cv2.inRange(img_YCrCb , lower_ycrcb , upper_ycrcb)

    lower_hsv = np.array([0,100,0] , dtype=np.uint8)
    upper_hsv = np.array([40,255,255] , dtype=np.uint8)
    skin_mask_hsv= cv2.inRange(img_hsv , lower_hsv , upper_hsv)


    skin_mask = cv2.bitwise_and(skin_mask_hsv , skin_mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
    skin_mask = cv2.morphologyEx(skin_mask , cv2.MORPH_CLOSE,kernel , iterations=1)

    skin_rgb = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB)

    return skin_rgb


def find_range_YCrCb(ycrcb):
    hist_y = cv2.calcHist([ycrcb], [0], None, [256], [0, 256])
    hist_cr = cv2.calcHist([ycrcb], [1] , None , [256] , [0 , 256])
    hist_cb = cv2.calcHist([ycrcb], [2] , None , [256] , [0 , 256])

    max_y = np.argmax(hist_y)
    print(max_y)
    if max_y < 200 and max_y > 100:
        lower_y = np.array([180,0,115] ,dtype=np.uint8)
        upper_y = np.array([255,200,130],dtype=np.uint8)
        return lower_y , upper_y
    
    else:
        lower_y = np.array([95,150,65],dtype=np.uint8)
        upper_y = np.array([255,230,130],dtype=np.uint8)
        return lower_y , upper_y



cv2.imshow('color' , skin_detection('s'))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('color' , skin_detection('s'))
cv2.waitKey()
cv2.destroyAllWindows()







# this is simple function that indicates how it works ( main and logic part )
# import cv2
# import numpy as np


# def skin_detection(image_path):

#     img = cv2.imread('images/face2.png')
#     img_YCrCb = cv2.cvtColor(img , cv2.COLOR_BGR2YCrCb)
#     img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

#     lower_ycrcb = np.array([95,150,65] , dtype=np.uint8)
#     upper_ycrcb = np.array([255,230,130] , dtype=np.uint8)
#     # lower_ycrcb = np.array([180,0,115] , dtype=np.uint8)
#     # upper_ycrcb = np.array([255,200 ,130] , dtype=np.uint8)
#     skin_mask_ycrcb = cv2.inRange(img_YCrCb , lower_ycrcb , upper_ycrcb)

#     lower_hsv = np.array([0,100,0] , dtype=np.uint8)
#     upper_hsv = np.array([40,255,255] , dtype=np.uint8)
#     skin_mask_hsv= cv2.inRange(img_hsv , lower_hsv , upper_hsv)


#     skin_mask = cv2.bitwise_and(skin_mask_hsv , skin_mask_ycrcb)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
#     skin_mask = cv2.morphologyEx(skin_mask , cv2.MORPH_CLOSE,kernel , iterations=1)

#     skin_rgb = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB)

#     return skin_rgb