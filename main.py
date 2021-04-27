import sys

from config import *
from meanShift import calculatePoints
from utils import *


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
    track_window = faces[0]

    # Initialize Spotlight and Party objects
    spot = Spotlight(frame)
    party_filter = Party()

    # Main loop
    while True:
        # Get next frame if there is one
        ret, frame = cap.read()
        if ret:
            # Convert to HSV and compute skin tone mask for the current frame
            dst = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.array((0., 60., 32.)),
                              np.array((180., 255., 255.)))

            # Run meanshift for the specified number of iterations
            for i in range(NUM_ITERATIONS):
                pts, track_window = calculatePoints(track_window, dst)

            # Allow intra-pixel computation
            frame = frame.astype(np.double)

            # Generate circular mask
            mask = calc_circle_mask(frame, track_window, SPOTLIGHT_RADIUS)

            # Darken background
            background = np.where(mask != 255)
            frame[background[0], background[1], :] -= 150

            # Overlay spotlight image
            imgr, imgc, spotr, spotc = spot.calc(track_window)
            frame[imgr, imgc, :] = spot.spotlight_rgb[spotr, spotc, :3] * 255

            # Apply party mode
            foreground = np.where(mask == 255)
            frame[foreground[0], foreground[1], party_filter.current_channel] += party_filter.get_current_color()
            party_filter.next_color()

            # Fix under/overflow
            frame[frame < 0] = 0
            frame[frame > 255] = 255
            frame = np.uint8(frame)

            # Display the current frame
            cv2.imshow('The Spotlight\'s On You', frame)

            # Exit loop if escape key is pressed
            if cv2.waitKey(DELAY) & 0xff == 27:
                break

        else:
            break

    # Cleanup
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
