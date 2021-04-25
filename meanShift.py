import numpy as np
import cv2
import time
import math
from copy import deepcopy


# using the bounding, choose a circle of region of interest
# use the circle of interest as indexing into dst for points in cicle
# find center of mass of dst indexed by the circel of interest
# shift bounding box to be centered on mean
def calculatePoints(trackWindow, radiusCircleInterest, dst):
    c,r,w,h = trackWindow[0],trackWindow[1] ,trackWindow[2] ,trackWindow[3]
    center = (c + w/2, r + h/2)
    #boundingBox = makeBoundingBox(track_window)

    #keep these as floats or integer overflow
    massCenterRow = 0.0
    massCenterCol = 0.0
    totalIntensity = 0.0
    #out = []

    # for col in range(c,c+w):
    #     for row in range(r,r+h):
    #
    # # for col in range(int(c + w/4),int(c+w - w/4)):
    # #     for row in range(int(r + h/4),int(r+h - h/4)):
    #         #out.append((row,col,dst[row][col]))
    #         if row >= len(dst):
    #             row = len(dst)-1
    #         elif col >= len(dst[0]):
    #             col = len(dst[0])-1
    #         massCenterRow += dst[row][col] * row
    #         massCenterCol += dst[row][col] * col
    #         totalIntensity += dst[row][col]
    # massCenterRow /= totalIntensity
    # massCenterCol /= totalIntensity

    intensities = dst[int(r):int(r+h),int(c):int(c+w)]
    grid = np.indices((dst.shape[0], dst.shape[1]))
    grid = grid[:,int(r):int(r+h),int(c):int(c+w)]

    massCenterRow = grid[0] * intensities
    massCenterCol = grid[1] * intensities

    massCenterRow = np.sum(np.ndarray.flatten(massCenterRow), dtype=np.int64)
    massCenterCol = np.sum(np.ndarray.flatten(massCenterCol), dtype=np.int64)
    #dtype has to be np.int64 to avoid overflow

    totalIntensity = np.sum(np.ndarray.flatten(intensities))
    massCenterRow = massCenterRow / totalIntensity
    massCenterCol = massCenterCol / totalIntensity

    if np.isnan(massCenterRow): massCenterRow = trackWindow[1]
    if np.isnan(massCenterCol): massCenterCol = trackWindow[0]

    massCenterRow = int(massCenterRow)
    massCenterCol = int(massCenterCol)


    pts = np.array([[massCenterCol-w/2, massCenterRow-h/2],
                    [massCenterCol-w/2, massCenterRow+h/2],
                    [massCenterCol+w/2, massCenterRow+h/2],
                    [massCenterCol+w/2, massCenterRow-h/2]])

    trackWindow=(massCenterCol-w/2, massCenterRow-h/2,w,h)
    return pts, trackWindow