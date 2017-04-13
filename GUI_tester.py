# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui, uic
import sys
import cv2
import numpy as np
import threading
import time
import Queue
import PIL
from PIL import Image, ImageDraw, ImageFont

cascPath = 'Cascades/merge_cascade_updated.xml'
cascPath2 = 'Cascades/added_lane_cascade_updated.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stop_cascade.xml'
cascPath6 = 'Cascades/stopAhead_cascade.xml'
cascPath7 = 'Cascades/signal_ahead_cascade.xml'
faceCasc = 'haarcascade_frontalface_default.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
stopAheadCascade = cv2.CascadeClassifier(cascPath6)
signalAheadCascade = cv2.CascadeClassifier(cascPath7)
faceCascade = cv2.CascadeClassifier(faceCasc)

# 0: English
# 1: Japanese
# 2: Spanish
# 3: French
language = 0
mergeText = ["Merge", unicode("マージ","utf-8"), unicode("Unir","utf-8"), unicode("fusionner","utf-8")]
addedLanesText = ["Added Lane", unicode("追加されたレーン","utf-8"), unicode("Carril añadido","utf-8"), unicode("Voies ajoutées","utf-8")]
pedestrianText = ["Pedestrian Crossing", unicode("横断歩道","utf-8"), unicode("cruce peatonal","utf-8"), unicode("passage piéton","utf-8")]
laneEndsText = ["Lane Ends", unicode("レーンエンド","utf-8"), unicode("Carril termina","utf-8"), unicode("La voie se termine","utf-8")]
stopText = ["Stop", unicode("やめる","utf-8"), unicode("Pare","utf-8"), unicode("Arrêtez","utf-8")]
stopAheadText = ["Stop Ahead", unicode("この先、一旦停止","utf-8"), unicode("Pare a continuación","utf-8"), unicode("Arrêt devant","utf-8")]
signalAheadText = ["Signal Ahead", unicode("この先、信号有り","utf-8"), unicode("Señal Adelante","utf-8"), unicode("signal devant","utf-8")]


running = False
capture_thread = None
form_class = uic.loadUiType("Resources\simple.ui")[0]
q = Queue.Queue()
 

def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print queue.qsize()
			
def cascadeUS(frame):
	global language
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	merge = mergeCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(20, 20),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	addedLanes = addedLaneCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	pedestrians = pedestrianCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	laneEnds = laneEndsCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	stop = stopCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	stopAhead = stopAheadCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	 
	signalAhead = signalAheadCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	for (x, y, w, h) in merge:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		frame = convertToLanguage(frame, (x, y), mergeText[language], (0, 0, 255))	
		#cv2.putText(frame, mergeText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255) )

	for (x, y, w, h) in addedLanes:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		frame = convertToLanguage(frame, (x, y), addedLanesText[language], (0, 255, 0))
		#cv2.putText(frame, addedLanesText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0)     )

	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		frame = convertToLanguage(frame, (x, y), pedestrianText[language], (255, 0, 0))
		#cv2.putText(frame, pedestrianText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0) )

	for (x, y, w, h) in laneEnds:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
		frame = convertToLanguage(frame, (x, y), laneEndsText[language], (255, 0, 255))
		#cv2.putText(frame, laneEndsText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255) )

	for (x, y, w, h) in stop:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		frame = convertToLanguage(frame, (x, y), stopText[language], (0, 255, 255))
		#cv2.putText(frame, stopText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255) )

	for (x, y, w, h) in stopAhead:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
		frame = convertToLanguage(frame, (x, y), stopAheadText[language], (255, 255, 0))
		#cv2.putText(frame, stopAheadText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0) )
		
	for (x, y, w, h) in signalAhead:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
		frame = convertToLanguage(frame, (x, y), signalAheadText[language], (0, 255, 100))
		#cv2.putText(frame, signalAheadText[language], (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 100) )
		
	return frame
	
def convertToJapanese(frame, (x, y), phrase, color):
	im = PIL.Image.fromarray(frame)
	draw = ImageDraw.Draw(im)
	font = ImageFont.truetype("Fonts/sazanami-mincho.ttf", 40)
	draw.text((x,y-40), phrase, font = font, fill=color)
	return np.asarray(im)
	
def convertToLanguage(frame, (x, y), phrase, color):
	global language
	if (language == 1):
		return convertToJapanese(frame, (x,y), phrase, color)
	else:
		im = PIL.Image.fromarray(frame)
		draw = ImageDraw.Draw(im)
		font = ImageFont.truetype("Fonts/AbhayaLibre-Regular.ttf", 40)
		draw.text((x,y-40), phrase, font = font, fill=color)
		return np.asarray(im)

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()



class MyWindowClass(QtGui.QMainWindow, form_class):
	def __init__(self, parent=None):
		QtGui.QMainWindow.__init__(self, parent)
		self.setupUi(self)
		self.EnglishButton.clicked.connect(self.select_English)
		self.JapaneseButton.clicked.connect(self.select_Japanese)
		self.SpanishButton.clicked.connect(self.select_Spanish)
		self.FrenchButton.clicked.connect(self.select_French)
		self.startButton.clicked.connect(self.start_clicked)
		
		self.window_width = self.ImgWidget.frameSize().width()
		self.window_height = self.ImgWidget.frameSize().height()
		self.ImgWidget = OwnImageWidget(self.ImgWidget)       

		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(1)


	def start_clicked(self):
		global running
		running = True
		capture_thread.start()
		self.startButton.setEnabled(False)
		self.startButton.setText('Starting...')
		
	def select_English(self):
		global language
		language = 0
	
	def select_Japanese(self):
		global language
		language = 1
		
	def select_Spanish(self):
		global language
		language = 2
		
	def select_French(self):
		global language
		language = 3

	def update_frame(self):
		if not q.empty():
			self.startButton.setText('Camera is live')
			frame = q.get()
			img = frame["img"]

			img_height, img_width, img_colors = img.shape
			scale_w = float(self.window_width) / float(img_width)
			scale_h = float(self.window_height) / float(img_height)
			scale = min([scale_w, scale_h])

			if scale == 0:
				scale = 1
			
			img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
			img = cascadeUS(img)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			height, width, bpc = img.shape
			bpl = bpc * width
			image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
			self.ImgWidget.setImage(image)

	def closeEvent(self, event):
		global running
		running = False



capture_thread = threading.Thread(target=grab, args = (0, q, 1920, 1080, 30))

app = QtGui.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('CS 472 (Group 5) Sign Detection Application')
w.show()
app.exec_()
