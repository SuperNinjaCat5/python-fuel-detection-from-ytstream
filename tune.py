import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera didn't open. Try VideoCapture(1), or close other apps using the camera.")
    raise SystemExit

cv2.namedWindow("tune", cv2.WINDOW_NORMAL)
cv2.resizeWindow("tune", 500, 300)
for name, maxv, start in [("H low",179,20),("S low",255,100),("V low",255,150),
                          ("H high",179,40),("S high",255,255),("V high",255,255)]:
    cv2.createTrackbar(name, "tune", start, maxv, lambda x: None)

blank = np.zeros((10, 500, 3), dtype="uint8")  # gives the tune window something to draw

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read a frame from the camera.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([cv2.getTrackbarPos(n, "tune") for n in ("H low","S low","V low")])
    hi = np.array([cv2.getTrackbarPos(n, "tune") for n in ("H high","S high","V high")])
    mask = cv2.inRange(hsv, lo, hi)

    cv2.imshow("tune", blank)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("detected", cv2.bitwise_and(frame, frame, mask=mask))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("lower:", lo, "upper:", hi)
        break

cap.release()
cv2.destroyAllWindows()