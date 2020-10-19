import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    # ...

    if ch == ord('q'):
        break

    if mode == 1:
        # just example code
        # your code should implement
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Display the resulting frame
    cv2.imshow('frame', frame)






# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()