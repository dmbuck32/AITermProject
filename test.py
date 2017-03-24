# import the necessary packages
#from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tk
import threading
import datetime
import imutils
import cv2
import os
import sys
import numpy as np
import argparse
import time
from imutils.video import VideoStream

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

class PhotoBoothApp:
	def __init__(self, vs):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tk.Tk()
		self.panel = None
		
		# create a button, that when pressed, will take the current
		# frame and save it to file
		#btn = tk.Button(self.root, text="Snapshot!")
			
		m1 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m1.pack(side="left", fill="both", expand="yes", padx=10, pady=10)	
		m1.set(1.5)
		m2 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m2.set(1.5)
		m2.pack(side="left", fill="both", expand="yes", padx=10, pady=10)	
		m3 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m3.set(1.5)
		m3.pack(side="left", fill="both", expand="yes", padx=10, pady=10)	
		m4 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m4.set(1.5)
		m4.pack(side="left", fill="both", expand="yes", padx=10, pady=10)	
		m5 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m5.set(1.5)
		m5.pack(side="left", fill="both", expand="yes", padx=10, pady=10)	
		m6 = tk.Scale(self.root, from_=1.01, to=1.99, resolution=.01)
		m6.set(1.5)
		m6.pack(side="left", fill="both", expand="yes", padx=10, pady=10)
		
		#btn.pack(side="left", fill="both", expand="yes", padx=10, pady=10)

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=(m1, m2, m3, m4, m5, m6))
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Computer Vision App")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		
	def videoLoop(self, m1, m2, m3, m4, m5, m6):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=300)
				
				self.cascade(self.frame, m1, m2, m3, m4, m5, m6)
		
				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
		
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tk.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image

		except RuntimeError, e:
			print("[INFO] caught a RuntimeError")
	
	def cascade(self, frame, m1, m2, m3, m4, m5, m6):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		merge = mergeCascade.detectMultiScale(
			gray,
			scaleFactor=m1.get(),
			minNeighbors=5,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
		)

		addedLanes = addedLaneCascade.detectMultiScale(
			gray,
			scaleFactor=m2.get(),
			minNeighbors=5,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
		)

		pedestrians = pedestrianCascade.detectMultiScale(
			gray,
			scaleFactor=m3.get(),
			minNeighbors=5,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
		)

		laneEnds = laneEndsCascade.detectMultiScale(
			gray,
			scaleFactor=m4.get(),
			minNeighbors=5,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
		)
		
		stop = stopCascade.detectMultiScale(
			gray,
			scaleFactor=m5.get(),
			minNeighbors=800,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
		)
		
		stopAhead = stopAheadCascade.detectMultiScale(
			gray,
			scaleFactor=m6.get(),
			minNeighbors=5,
			minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE
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
			
	'''
	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))

		# save the file
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))
	'''	
	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()
		self.root.destroy()
		
	def nothing(x):
		pass

def nothing(x):
    pass
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	'''
	ap.add_argument("-o", "--output", required=True,
		help="path to output directory to store snapshots")
	ap.add_argument("-p", "--picamera", type=int, default=-1,
		help="whether or not the Raspberry Pi camera should be used")
	'''
	args = vars(ap.parse_args())

	# initialize the video stream and allow the camera sensor to warmup
	#print "[INFO] warming up camera..."
	vs = VideoStream(0).start()
	#time.sleep(2.0)

	# start the app
	pba = PhotoBoothApp(vs)
	pba.root.mainloop()	
	
def main2():
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

# Run the script
if __name__ == "__main__":
   #main2()
   main()
   sys.exit()