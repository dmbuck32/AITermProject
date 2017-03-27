import cv2
import sys
import numpy as np

cascPath = 'cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)


	# define range of blue color in HSV
	yellow_lower = np.array([20,90,80])
	yellow_upper = np.array([90,250,200])

	blue_lower = np.array([120, 50, 50])
	blue_upper = np.array([170, 255, 230])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, blue_lower, blue_upper)
	mask2 = cv2.inRange(hsv, yellow_lower, yellow_upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)
	res2 = cv2.bitwise_and(frame, frame, mask = mask2)

	cv2.imshow('frame',frame)
	cv2.imshow('mask', mask)
	cv2.imshow('mask2', mask2)
	cv2.imshow('res', res)
	cv2.imshow('res2', res2)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
