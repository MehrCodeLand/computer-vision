import cv2
import numpy as np

# part one is about face detection
face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

img = cv2.imread('images/face.jpg')
gray  =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.3,5)

if faces is ():
    print('No faces detected')

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('face',img )
    cv2.waitKey(0)

cv2.destroyAllWindows()


