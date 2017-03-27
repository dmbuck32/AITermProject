import cv2
import sys
import numpy as np

#Classifiers
cascPath = 'Cascades/merge_cascade.xml'
cascPath2 = 'Cascades/added_lane_cascade.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stopCascade.xml'
cascPath6 = 'Cascades/stopAhead_cascade.xml'
cascPath7 = 'Cascades/signal_ahead_cascade.xml'
cascPath8 = 'Cascades/diamond_sign_cascade.xml'

mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
stopAheadCascade = cv2.CascadeClassifier(cascPath6)
signalAheadCascade = cv2.CascadeClassifier(cascPath7)
diamondSignCascade = cv2.CascadeClassifier(cascPath8)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    
    diamondSigns = diamondSignCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
#    stop = stopCascade.detectMultiScale(
#        gray,
#        scaleFactor=1.1,
#        minNeighbors=5,
#        minSize=(30, 30),
#        flags=cv2.CASCADE_SCALE_IMAGE
#    )
    
    for (x, y, w, h) in diamondSigns:
        cv2.rectangle(frame, (x - 10, y - 10), (x+w+20, y+h+20), (0, 255, 0), 2)
        roi = hsv[y-10:y+h+20, x-10:x+w+h+20]
        
#        merge = mergeCascade.detectMultiScale(
#            roi,
#            scaleFactor=1.1,
#            minNeighbors=5,
#            minSize=(30, 30),
#            flags=cv2.CASCADE_SCALE_IMAGE
#        )
#        for (x, y, w, h) in merge:
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#        
#        addedLane = addedLaneCascade.detectMultiScale(
#            roi,
#            scaleFactor=1.1,
#            minNeighbors=5,
#            minSize=(30, 30),
#            flags=cv2.CASCADE_SCALE_IMAGE
#        )
#        for (x, y, w, h) in addedLane:
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 102), 2)
        
        
        pedestrian = pedestrianCascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x2, y2, w, h) in pedestrian:
            cv2.rectangle(roi, (x+x2, y+y2), (x2+w+x, y2+h+y), (255, 0, 102), 2)
#        
#        
#        laneEnds = laneEndsCascade.detectMultiScale(
#            roi,
#            scaleFactor=1.1,
#            minNeighbors=5,
#            minSize=(30, 30),
#            flags=cv2.CASCADE_SCALE_IMAGE
#        )
#        for (x, y, w, h) in laneEnds:
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 153, 0), 2)
#        
#        
#        stopAhead = stopAheadCascade.detectMultiScale(
#            roi,
#            scaleFactor=1.1,
#            minNeighbors=5,
#            minSize=(30, 30),
#            flags=cv2.CASCADE_SCALE_IMAGE
#        )
#        for (x, y, w, h) in stopAhead:
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        
#        signalAhead = signalAheadCascade.detectMultiScale(
#            roi,
#            scaleFactor=1.1,
#            minNeighbors=5,
#            minSize=(30, 30),
#            flags=cv2.CASCADE_SCALE_IMAGE
#        )
#        for (x2, y2, w, h) in signalAhead:
#            cv2.rectangle(frame, (x+x2, y+y2), (x2+x+w, y2+y+h), (255, 255, 255), 2)
#            
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
        
        
    
        
    
    
    