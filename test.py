import cv2, sys
import numpy as np
import Tkinter as tk

def nothing(x):
    pass
	
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

# Create a black image, a window
#img = np.zeros((len(frame), len(frame[0])), np.uint8)
cv2.namedWindow('image')

# create switch for ON/OFF functionality
switch = 'Video Feed'
cv2.createTrackbar(switch, 'image',1,1,nothing)

while True:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret2,thresh = cv2.threshold(gray, 127,255,0)
	im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img = cv2.drawContours(im, contours, -1, (0,255,0), 1)
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF
	if k == 27: #escape
		break

	# get current positions of four trackbars
	s = cv2.getTrackbarPos(switch,'image')

	if s == 1:
		ret, frame = video_capture.read()		

video_capture.release()
cv2.destroyAllWindows()

