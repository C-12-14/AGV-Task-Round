from dehazing import Dehazing
import time
import cv2
import os

dirname = os.path.dirname(__file__)


def dehaze_video():
    dehazer = Dehazing(delta=0.4, sigma=300, brightness_gain=1.1, kernel=(65, 65))
    cap = cv2.VideoCapture(os.path.join(dirname, "Hazy/hazy.mp4"))
    out = cv2.VideoWriter(
        os.path.join("Dehazed/dehazed23.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (1920, 1080),
    )
    newft = 0
    prevft = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        newft = time.time()
        fps = 1 / (newft - prevft)
        prevft = newft
        fps = int(fps)
        frame = dehazer.dehaze(frame)

        cv2.putText(frame, str(fps), (1, 50), font, 1.5, (100, 255, 0), 2, cv2.LINE_AA)
        frame = cv2.resize(frame, (int(frame.shape[1] * 2), int(frame.shape[0] * 2)))
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


dehaze_video()
