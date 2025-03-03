import numpy as np 
import cv2 

# convert to binary image
img = cv2.imread('images/red.png')
img = cv2.resize(img ,(300,300) )
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
_ , binary_image = cv2.threshold(gray , 128 , 255 , cv2.THRESH_BINARY)
pixels = binary_image.flatten()


def rle_encode(pixels):
    values = []
    counts = []
    prev_pixel = pixels[0]
    count = 1
    
    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            values.append(prev_pixel)
            counts.append(count)
            prev_pixel = pixel
            count = 1
    
    values.append(prev_pixel)
    counts.append(count)
    
    return values, counts

def rle_decode(values, counts, shape):
    pixels = np.concatenate([np.full(c, v, dtype=np.uint8) for v, c in zip(values, counts)])
    return pixels.reshape(shape)





# encode the image
values , counts = rle_encode(pixels)

# show original image first 
cv2.imshow('orginal binary' , binary_image)
cv2.waitKey()

# decode image (numbers)
decoded_image = rle_decode(values, counts, binary_image.shape)
cv2.imshow('decode' , decoded_image)
cv2.waitKey()
cv2.destroyAllWindows()



# finding diffrent space 
import sys 
print(sys.getsizeof(counts))
print(sys.getsizeof(values))
print(sys.getsizeof(pixels))