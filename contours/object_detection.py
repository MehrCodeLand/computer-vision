import cv2
shape = ''

image = cv2.imread('rec.png')
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150,255 , cv2.THRESH_BINARY_INV)

contours , _ = cv2.findContours(thresh , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue
    
    epsilon = 0.04 * cv2.arcLength(contour , True )
    approx = cv2.approxPolyDP( contour , epsilon , True )
    
    if len(approx)==3 :
        shape = 'rectangle'
    elif len(approx) == 4 :
        x, y , w ,h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            shape = 'Square'
        else:
            shape = 'Rectangle'
    elif len(approx) > 4 :
        shape = 'Circle'
    else:
        shape = 'Unknown'
        
    
    cv2.drawContours(image , [contour] , -1 , (0,255,0) , 2 )
    cv2.putText(image , shape , (contour[0][0][0] , contour[0][0][1] - 10 ) , cv2.FONT_HERSHEY_COMPLEX , 0.5 , (0,255,0) , 2)
    
    
     # Display the image with contours and labels
     
cv2.imshow('Shape Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
