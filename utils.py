import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_faces(frame):
    """Detects a face given a frame of video"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return detector.detectMultiScale(gray, 1.1, 4)


def calc_circle_mask(frame, track_window, radius=125):
    """Calculates circular mask given a bounding box"""
    c, r, w, h = track_window
    r, c, h, w = int(r), int(c), int(h), int(w)

    center = (np.float32(c + int(w / 2)), np.float32(r + int(h / 2)))
    color = (0, 255, 255)
    thickness = 2
    cv2.circle(frame, center, radius, color, thickness)

    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    return mask


class Spotlight:
    """
    Object to overlay a spotlight to the top of the screen that follows a moving point
    """

    def __init__(self, frame):
        # Spotlight Things
        spotlight_raw_bgr = plt.imread("spotlight.png")
        spotlight_raw_rgb = cv2.cvtColor(spotlight_raw_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        spot_w = int(frame_w / 7)
        spot_h = int(spotlight_raw_rgb.shape[0] * spot_w / spotlight_raw_rgb.shape[1])
        self.spot_pivot = np.array([int(spot_w * (15 / 64)), int(spot_h * (15 / 87)), 1])

        self.glob_spot_start = np.array(
            [int((frame_w / 2) - self.spot_pivot[0]), int((frame_h / 15) - self.spot_pivot[1]), 1])
        self.glob_spot_mid = np.array([int(frame_w / 2), int(frame_h / 12), 1])

        self.spotlight_rgb = cv2.resize(spotlight_raw_rgb, (spot_w, spot_h))
        spotlight_bgr = cv2.resize(spotlight_raw_bgr, (spot_w, spot_h))

        self.spot_points = np.ones((1, 3))

        for i in range(spot_h):
            for j in range(spot_w):
                if spotlight_bgr[i, j, 3] != 0:
                    point = np.array([[j, i, 1]])  # we put this in an x, y, coordinate
                    self.spot_points = np.vstack((self.spot_points, point))

        self.spot_points = np.delete(self.spot_points, 0, 0)

    def calc(self, track_window):
        c, r, w, h = track_window
        # Start spot
        face_mid = np.array([int(c + w / 2), int(r + h / 2), 1])
        dif = self.glob_spot_mid - face_mid
        rotv = -math.atan2(dif[1], dif[0]) - 2.05

        rot = np.array([[np.cos(rotv), -np.sin(rotv), 0], [np.sin(rotv), np.cos(rotv), 0], [0, 0, 1]])
        spot_points_rot = self.spot_points @ rot
        spot_pivot_new = self.spot_pivot @ rot
        spot_pivot_diff = spot_pivot_new - self.spot_pivot

        imgr = spot_points_rot[:, 1].astype(np.int32) + int(self.glob_spot_start[1]) - int(spot_pivot_diff[1])
        imgc = spot_points_rot[:, 0].astype(np.int32) + int(self.glob_spot_start[0]) - int(spot_pivot_diff[0])
        spotr = self.spot_points[:, 1].astype(np.int32)
        spotc = self.spot_points[:, 0].astype(np.int32)

        return imgr, imgc, spotr, spotc

    glob_spot_start = None
    glob_spot_mid = None
    spotlight_rgb = None
    spot_points = None
    spot_pivot = None


class Party:
    """
    Multicolored filter handler for video overlay
    """

    def __init__(self):
        self.color_count = 0
        self.current_color = np.linspace(0, 100)
        self.current_channel = 0

    def next_color(self):
        if self.color_count == len(self.current_color) - 1:
            self.color_count = 0
            if self.current_channel == 2:
                self.current_channel = 0
            else:
                self.current_channel += 1
        else:
            self.color_count += 1

    def get_current_color(self):
        return self.current_color[self.color_count]

    color_count = None
    current_color = None
    current_channel = None
