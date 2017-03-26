import cv2
import sys
import numpy as np

cascPath = 'Cascades/signal_ahead_cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    
    yellow_lower = np.array([20,90,80])
    yellow_upper = np.array([90,250,200])
    
    blue_lower = np.array([120, 50, 50])
    blue_upper = np.array([170, 255, 230])

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = hsv[y:y+h, x:x+w]
        mask = cv2.inRange(roi, yellow_lower, yellow_upper)
        res = cv2.bitwise_and(roi, roi, mask= mask)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    try:
        mask
    except NameError:
        break
    else:
        cv2.imshow('Mask', mask)
        cv2.imshow('res',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
