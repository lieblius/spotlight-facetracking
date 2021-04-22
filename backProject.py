import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np

def calcBackProject(image, roi_hist):
    hsvt = image

    M = roi_hist
    I = cv2.calcHist([hsvt], [0], None, [180], [0, 180])

    R = M/I
    h, s, v = cv2.split(hsvt)

    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsvt.shape[:2])

