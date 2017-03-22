import cv2, sys

cascPath = 'Cascades/merge_cascade.xml'
cascPath2 = 'Cascades/added_lane_cascade.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stopCascade.xml'
cascPath6 = 'Cascades/stopAhead_cascade.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
stopAheadCascade = cv2.CascadeClassifier(cascPath6)

video_capture = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	merge = mergeCascade.detectMultiScale(
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

	laneEnds = laneEndsCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	
	stop = stopCascade.detectMultiScale(
		gray,
		scaleFactor=1.4,
		minNeighbors=800,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	
	stopAhead = stopAheadCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
	   flags=cv2.CASCADE_SCALE_IMAGE
	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in merge:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.putText(frame, "Merge", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255) )
		
	for (x, y, w, h) in addedLanes:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, "Added Lanes", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0)	 )

	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv2.putText(frame, "Pedestrian Crossing", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0) )
		
	for (x, y, w, h) in laneEnds:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
		cv2.putText(frame, "Lane Ends", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255) )
	
	for (x, y, w, h) in stop:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv2.putText(frame, "Stop", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255) )
	
	for (x, y, w, h) in stopAhead:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
		cv2.putText(frame, "Stop Ahead", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0) )
		
	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
