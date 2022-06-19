from dehazing import Dehazing
import time
import cv2


def dehaze_video():
    dehazer = Dehazing()
    cap = cv2.VideoCapture("Image-Dehazing/Hazy/hazy.mp4")
    out = cv2.VideoWriter(
        "Image-Dehazing/Dehazed/dehazed.mp4",
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
        cv2.putText(frame, str(fps), (1, 50), font, 1.5, (100, 255, 0), 2, cv2.LINE_AA)
        frame = dehazer.dehazing(frame, 0.8, sigma=7, bgain=1.02)
        frame = cv2.resize(frame, (int(frame.shape[1] * 2), int(frame.shape[0] * 2)))
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


dehaze_video()
