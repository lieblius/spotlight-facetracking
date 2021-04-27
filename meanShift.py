import numpy as np


def calculatePoints(trackWindow, dst):
    """
    Mean-shift algorithm. Using the bounding box, chooses a circle of region of interest and uses the circle of
    interest as indexing into dst for points in cicle. Finds center of mass of dst indexed by the circel of interest
    and shifts bounding box to be centered on mean.
    """
    c, r, w, h = trackWindow[0], trackWindow[1], trackWindow[2], trackWindow[3]

    intensities = dst[int(r):int(r + h), int(c):int(c + w)]
    grid = np.indices((dst.shape[0], dst.shape[1]))
    grid = grid[:, int(r):int(r + h), int(c):int(c + w)]

    massCenterRow = grid[0] * intensities
    massCenterCol = grid[1] * intensities

    # dtype has to be np.int64 to avoid overflow
    massCenterRow = np.sum(np.ndarray.flatten(massCenterRow), dtype=np.int64)
    massCenterCol = np.sum(np.ndarray.flatten(massCenterCol), dtype=np.int64)

    totalIntensity = np.sum(np.ndarray.flatten(intensities))
    massCenterRow = np.divide(massCenterRow, totalIntensity)
    massCenterCol = np.divide(massCenterCol, totalIntensity)

    if np.isnan(massCenterRow): massCenterRow = trackWindow[1]
    if np.isnan(massCenterCol): massCenterCol = trackWindow[0]

    massCenterRow = int(massCenterRow)
    massCenterCol = int(massCenterCol)

    pts = np.array([[massCenterCol - w / 2, massCenterRow - h / 2],
                    [massCenterCol - w / 2, massCenterRow + h / 2],
                    [massCenterCol + w / 2, massCenterRow + h / 2],
                    [massCenterCol + w / 2, massCenterRow - h / 2]])

    trackWindow = (massCenterCol - w / 2, massCenterRow - h / 2, w, h)
    return pts, trackWindow
