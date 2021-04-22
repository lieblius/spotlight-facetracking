import numpy as np
import cv2
import time
import math
from copy import deepcopy
from meanShift import calculatePoints
import sys
import logging
from PIL import Image
import matplotlib.pyplot as plt


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
#cv2.imshow('img', frame2)
track_window = tuple(face[0])

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
#cv2.imshow('roi', roi)
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
#cv2.imshow('mask', mask)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# for x, y in dst:
# if rad^2 >= (x-center[0])^2 + (y - center[1])^2:
#   add point to be counted in calculating the center of mass (r, c, intensity)

#Spotlight Things
spotlight_raw_bgr = plt.imread("spotlight.png")
spotlight_raw_rgb = cv2.cvtColor(spotlight_raw_bgr, cv2.COLOR_BGR2RGB)
frame_h, frame_w = frame.shape[0], frame.shape[1]
spot_w = int(frame_w / 7)
spot_h = int(spotlight_raw_rgb.shape[0] * spot_w / spotlight_raw_rgb.shape[1])
spot_pivot = np.array([int(spot_w * (15/64)), int(spot_h * (15/87)), 1])

glob_spot_start = np.array([int((frame_w / 2) - spot_pivot[0]), int((frame_h / 15) - spot_pivot[1]), 1])
glob_spot_mid = np.array([int(frame_w / 2), int(frame_h / 12), 1])

spotlight_rgb = cv2.resize(spotlight_raw_rgb, (spot_w, spot_h))
spotlight_bgr = cv2.resize(spotlight_raw_bgr, (spot_w, spot_h))

rotv = -0.42
rot= np.array([[np.cos(rotv), -np.sin(rotv), 0], [np.sin(rotv), np.cos(rotv), 0], [0, 0, 1]])
spot_points = np.ones((1,3))

for i in range(spot_h):
    for j in range(spot_w):
        if(spotlight_bgr[i,j,3] != 0):
            point = np.array([[j, i, 1]]) #we put this in an x, y, coordinate
            spot_points = np.vstack((spot_points, point))

spot_points = np.delete(spot_points, 0, 0)
spot_points_rot = spot_points @ rot
spot_pivot_new = spot_pivot @ rot


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

        c, r, w, h = track_window
        #Start spot
        face_mid = np.array([int(c + w/2), int(r + h/2), 1])
        dif = glob_spot_mid - face_mid
        rotv = -math.atan2(dif[1], dif[0]) - 2.05
        
        rot = np.array([[np.cos(rotv), -np.sin(rotv), 0], [np.sin(rotv), np.cos(rotv), 0], [0, 0, 1]])
        spot_points_rot = spot_points @ rot
        spot_pivot_new = spot_pivot @ rot
        spot_pivot_diff = spot_pivot_new - spot_pivot
        
        imgr = spot_points_rot[:,1].astype(np.int32) + int(glob_spot_start[1]) - int(spot_pivot_diff[1])
        imgc = spot_points_rot[:,0].astype(np.int32) + int(glob_spot_start[0]) - int(spot_pivot_diff[0])
        spotr = spot_points[:,1].astype(np.int32)
        spotc = spot_points[:,0].astype(np.int32)
        
        frame[imgr, imgc, :]  = spotlight_rgb[spotr, spotc, :3]*255
        
        # for i in range(spot_points_rot.shape[0]):
        #     imgr = int(spot_points_rot[i][1] + glob_spot_start[1] - spot_pivot_diff[1])
        #     imgc = int(spot_points_rot[i][0] + glob_spot_start[0] - spot_pivot_diff[0])
        #     spotr = int(spot_points[i][1])
        #     spotc = int(spot_points[i][0])
        #     frame[imgr, imgc, :]  = spotlight_rgb[spotr, spotc, :3]*255
        # #End spot    

        # frame /= 1.1
        # frame = int(frame)
        # #[r:r+h,c:c+w,:] /= 4
        frame = np.double(frame)

        frame = frame - 100
        frame[int(r):int(r+h),int(c):int(c+w),:] += 100
        frame[frame < 0] = 0
        frame = np.uint8(frame)

        # frameHSV  = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        # frameHSV = frameHSV[:,:,2] - 20
        # frameHSV[frameHSV < 0] = 0
        # frame = cv2.cvtColor(frameHSV, cv2.COLOR_HSV2BGR)
        #
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw it on image
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        #img2 = cv2.polylines(frame, np.int32([pts]), True, 255, 2)
        cv2.imshow('img2', frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", frame)

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
