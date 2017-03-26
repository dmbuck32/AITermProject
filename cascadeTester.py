import cv2, sys

cascPath = 'Cascades/merge_cascade.xml'
cascPath2 = 'Cascades/added_lane_cascade.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stopCascade.xml'
cascPath6 = 'Cascades/stopAhead_cascade.xml'
faceCasc = 'haarcascade_frontalface_default.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
stopAheadCascade = cv2.CascadeClassifier(cascPath6)
faceCascade = cv2.CascadeClassifier(faceCasc)

def nothing(self):
	pass

def main():
	video_capture = cv2.VideoCapture(0)
	ret, frame = video_capture.read()
	
	cv2.namedWindow('image')

	# create switch for ON/OFF functionality
	cv2.createTrackbar('Merge','image',50,98,nothing)
	cv2.createTrackbar('Added Lane','image',50,98,nothing)
	cv2.createTrackbar('Pedestrian','image',50,98,nothing)
	cv2.createTrackbar('Lane End','image',50,98,nothing)
	cv2.createTrackbar('Stop','image',50,98,nothing)
	cv2.createTrackbar('Stop Ahead','image',50,98,nothing)
	cv2.createTrackbar('Video Feed', 'image',1,2,nothing)
	
	while True:
		
		k = cv2.waitKey(1) & 0xFF
		if k == 27: #escape
			break

		# get current positions of four trackbars
		s = cv2.getTrackbarPos('Video Feed','image')
		c1 = cv2.getTrackbarPos('Merge','image')
		c2 = cv2.getTrackbarPos('Added Lane','image')
		c3 = cv2.getTrackbarPos('Pedestrian','image')
		c4 = cv2.getTrackbarPos('Lane End','image')
		c5 = cv2.getTrackbarPos('Stop','image')
		c6 = cv2.getTrackbarPos('Stop Ahead','image')

		if s == 1:
			ret, frame = video_capture.read()
			cascade(frame, c1, c2, c3, c4, c5, c6)
			cv2.imshow('image',frame)
		elif s == 2:
			ret, frame = video_capture.read()
			cv2.imshow('image',contour(frame))
			
def contour(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret2, thresh = cv2.threshold(gray, 127,255,0)
	im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img = cv2.drawContours(im, contours, -1, (0,255,0), 1)
	return img
			
def convert(in_position):
    in_position += 101
    in_position /= 100.0
    return in_position

def cascade(frame, c1, c2, c3, c4, c5, c6):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	merge = mergeCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c1),
		minNeighbors=5,
		minSize=(20, 20),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	addedLanes = addedLaneCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c2),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	pedestrians = pedestrianCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c3),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	laneEnds = laneEndsCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c4),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	stop = stopCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c5),
		minNeighbors=800,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	stopAhead = stopAheadCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c6),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

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
		
if __name__ == "__main__":
   main()
   sys.exit()