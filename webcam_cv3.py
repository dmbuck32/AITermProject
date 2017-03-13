import cv2
import sys

cascPath = 'merge_cascade.xml'
cascPath2 = 'added_lane_cascade.xml'
cascPath3 = 'pedestrianCrossing_cascade.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    merges = mergeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    addedLanes = addedLaneCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    pedestrians = pedestrianCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in merges:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    for (x, y, w, h) in addedLanes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
