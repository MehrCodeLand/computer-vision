import cv2
import numpy as np

def sift_detector(new_image, image_template):
    img1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()

    keypoint_1, descriptor_1 = sift.detectAndCompute(img1, None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches)

cap = cv2.VideoCapture(0)
image_template = cv2.imread("./images/DailyArt2.jpg")

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

    frame = cv2.flip(frame, 1)
    matches = sift_detector(cropped, image_template)
    cv2.putText(frame, str(matches), (450, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    threshold = 10

    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.putText(frame, 'Object Found', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
