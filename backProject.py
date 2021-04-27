import cv2
import numpy as np


def calcBackProject(hsvt, roi_hist):
    M = roi_hist
    I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

    R = M / I
    h, s, v = cv2.split(hsvt)

    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsvt.shape[:2])

    return B
