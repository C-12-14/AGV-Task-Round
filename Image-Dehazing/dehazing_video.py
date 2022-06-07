import numpy as np
import cv2
from dehazing_images import dehazing
import time


cap = cv2.VideoCapture("Image-Dehazing/hazy.mp4")
out = cv2.VideoWriter(
    "Image-Dehazing/dehazed2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
)
newft = 0
prevft = 0
font = cv2.FONT_HERSHEY_SIMPLEX
fpss = np.array([])
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
    newft = time.time()
    fps = 1 / (newft - prevft)
    prevft = newft
    fps = int(fps)
    # fpss = np.append(fpss, fps)
    cv2.putText(frame, str(fps), (1, 50), font, 1.5, (100, 255, 0), 2, cv2.LINE_AA)
    frame = dehazing(frame, 0.2, sigma=15)
    # cv2.imshow("frame", frame)
    frame = cv2.resize(frame, (int(frame.shape[1] * 2.5), int(frame.shape[0] * 2.5)))
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    out.write(frame)
print(fpss.mean())
cap.release()
out.release()
cv2.destroyAllWindows()
