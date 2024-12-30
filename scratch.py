import cv2
import numpy as np

# image = cv2.imread('photo_2024-12-27_14-36-51.jpg', 0) #photo_1_2024-12-27_15-07-35.jpg
image = cv2.imread('photo_2024-12-27_14-36-51.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
objects = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            objects.append((contour, (cx, cy)))


objects.sort(key=lambda x: (x[1][1], x[1][0]))

for i, (contour, center) in enumerate(objects):
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    cv2.putText(image, str(i+1), (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Image with Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
