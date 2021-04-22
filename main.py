import numpy as np
import cv2
import time
import math
from copy import deepcopy
from meanShift import calculatePoints


NUM_ITERAIONS = 10

# cap = cv2.VideoCapture('vid2.mp4')
cap = cv2.VideoCapture(0)

# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = detector.detectMultiScale(gray, 1.1, 4)
while len(face) == 0:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = detector.detectMultiScale(gray, 1.1, 4)
# r,h,c,w = 250,90,400,125  # simply hardcoded the values
# print(face)
(c, r, w, h) = face[0]
frame2 = deepcopy(frame)
cv2.rectangle(frame2, (c, r), (c + w, r + h), (255, 0, 0), 2)
cv2.imshow('img', frame2)
track_window = tuple(face[0])

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
cv2.imshow('roi', roi)
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
cv2.imshow('mask', mask)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# for x, y in dst:
# if rad^2 >= (x-center[0])^2 + (y - center[1])^2:
#   add point to be counted in calculating the center of mass (r, c, intensity)


while True:
    # time.sleep(1)
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1 )

        # apply meanshift to get the new location
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = 0
        for i in range(NUM_ITERAIONS):
            pts, track_window = calculatePoints(track_window, 10, dst)

        # Draw it on image
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        img2 = cv2.polylines(frame, np.int32([pts]), True, 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()


# #input track_window
# #output np array of points representing bounding box
#This might be wrong
def makeBoundingBox(track_window):
    return np.array([[track_window[0], track_window[1]],
                     [track_window[0]+ track_window[2], track_window[1]],
                     [track_window[0], track_window[1] + track_window[3]],
                     [track_window[0]+ track_window[2], track_window[1] + track_window[3]]])
