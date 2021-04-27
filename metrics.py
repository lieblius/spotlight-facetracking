import sys
from copy import deepcopy

from config import *
from meanShift import calculatePoints
from utils import *


def calc_iou(A, B):
    """
    Returns the intersection over union for two bounding boxes.
    Input is two tuples of x, y, w, h representing a bounding box
    """
    w = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    h = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    area = w * h
    return area / (A[2] * A[3] + B[2] * B[3] - area)


def main():
    # Read in video
    if len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[-1])
        if not cap.isOpened():
            print('Please enter a valid video path "python main.py <video_path>", or no path to capture from webcam.')
            quit(1)
    else:
        cap = cv2.VideoCapture(0)

    # Read frames until a face is detected
    frame = None
    faces = []
    while len(faces) == 0:
        _, frame = cap.read()
        faces = detect_faces(frame)
    (c, r, w, h) = faces[0]
    frame2 = deepcopy(frame)
    cv2.rectangle(frame2, (c, r), (c + w, r + h), (255, 0, 0), 2)

    # cv2.imshow('img', frame2)

    track_window = tuple(faces[0])
    track_window_2 = deepcopy(track_window)

    roi = frame[r:r + h, c:c + w]

    # cv2.imshow('roi', roi)

    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    # mask = cv2.bitwise_not(cv2.inRange(hsv_roi, np.array([10, 40, 100]), np.array([26, 70, 240])))

    cv2.imshow('mask', mask)

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # color = ('purple', 'blue', 'yellow')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([hsv_roi], [i], None, [256], [0, 256])
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # plt.imshow('roi_hist', roi_hist)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    total_iou = []
    running_iou = 0

    while True:
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # dst = calcBackProject(hsv, roi_hist)
            # mask = cv2.bitwise_not(cv2.inRange(hsv, np.array([10, 40, 100]), np.array([26, 70, 240])))
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            dst = mask

            for i in range(NUM_ITERATIONS):
                pts, track_window = calculatePoints(track_window, dst)
                ret, track_window_2 = cv2.meanShift(dst, track_window_2, term_crit)

            iou = calc_iou(track_window, track_window_2)
            total_iou.append(iou)
            running_iou = np.convolve(total_iou, np.ones(len(total_iou)) / len(total_iou), mode="valid").item()

            x1, y1, w1, h1 = np.int32(track_window)
            # frame = dst
            img = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            cv2.putText(img, 'ours', (x1, y1 + h1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            x2, y2, w2, h2 = track_window_2
            img = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            cv2.putText(img, 'opencv', (x2 + w - 100, y2 + h2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img,
                        f'IOU: {round(iou, 3)}',
                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img,
                        f'Running Avg IOU: {round(running_iou, 3)}',
                        (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow('metrics', img)

            k = cv2.waitKey(DELAY) & 0xff
            if k == 27:
                break

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    print(f'Average IOU: {round(running_iou, 5)}')
    plt.figure()
    plt.plot(total_iou)
    plt.ylim([0, 1])
    plt.xlabel("Frame #")
    plt.ylabel("IOU")
    plt.show()


if __name__ == '__main__':
    main()
