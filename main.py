import cv2
import numpy as np
import streamlink

url = "https://www.youtube.com/watch?v=3GE2fbbQkaU"

streams = streamlink.streams(url)
print("Available:", list(streams.keys()))  

stream_url = streams["480p"].url  
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("Couldn't open the stream")
    raise SystemExit

lowerC = np.array([20, 100, 150], dtype="uint8")
upperC = np.array([40, 255, 255], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    
    h, w = frame.shape[:2]
    my = h * 11 // 20

    frame[h * 5 // 8 :, w // 2 - w // 12 : w // 2 + w // 12] = 0 # vino al apagon ay dios
    halves = [frame[:my, :], frame[my:, :w // 2], frame[my:, w // 2:]]
    results = []
    areas = []

    for half in halves:
        original = half.copy()

        hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerC, upperC)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [c for c in cnts if cv2.contourArea(c) > 100]  

        area = 0
        for c in cnts:
            area += cv2.contourArea(c)
            cv2.drawContours(original, [c], 0, (0, 0, 0), 2)
        areas.append(area)

        overlay = original.copy()  
        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            M = cv2.moments(biggest)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                layer = overlay.copy()
                cv2.circle(layer, (cx, cy), 20, (0, 255, 0), -1)
                cv2.circle(layer, (cx, cy), 4, (0, 0, 255), -1)

                alpha = 0.65
                overlay = cv2.addWeighted(layer, alpha, overlay, 1 - alpha, 0)

        results.append(overlay)

    display = np.vstack((results[0], np.hstack((results[1], results[2]))))

    cv2.line(display, (0, my), (w, my), (255, 255, 255), 1)
    cv2.line(display, (w // 2, my), (w // 2, h), (255, 255, 255), 1)
    cv2.rectangle(display, (w // 2 - w // 12, h * 5 // 8), (w // 2 + w // 12, h - 1), (0, 0, 255), 1)
    cv2.imshow('overlay', display)
    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break
    if key == ord('f'):
        cv2.setWindowProperty('overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if key == ord('n'):
        cv2.setWindowProperty('overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()