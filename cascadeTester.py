# -*- coding: UTF-8 -*-
import cv2, sys, PIL
import numpy as np
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

def nothing(self):
    pass

def main():
    global language
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    
    cv2.namedWindow('image')

    # create switch for ON/OFF functionality
    cv2.createTrackbar('Merge','image',50,98,nothing)
    cv2.createTrackbar('Added Lane','image',50,98,nothing)
    cv2.createTrackbar('Pedestrian','image',50,98,nothing)
    cv2.createTrackbar('Lane End','image',50,98,nothing)
    cv2.createTrackbar('Signal Ahead','image',50,98,nothing)
    cv2.createTrackbar('Stop','image',50,98,nothing)
    cv2.createTrackbar('Stop Ahead','image',20,98,nothing)
    cv2.createTrackbar('Video Feed', 'image',1,4,nothing)
    cv2.createTrackbar('Language', 'image',0,3,nothing)
    
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
        c7 = cv2.getTrackbarPos('Signal Ahead','image')
        language = cv2.getTrackbarPos('Language','image')
        
        if s == 1:
            ret, frame = video_capture.read()
            cv2.imshow('image', cascadeUS(frame, c1, c2, c3, c4, c5, c6, c7))
        elif s == 2:
            ret, frame = video_capture.read()
            cv2.imshow('image', contour(frame))
        elif s == 3:
            ret, frame = video_capture.read()
            squares = find_squares(frame)
            cv2.drawContours( frame, squares, -1, (0, 255, 0), 3 )
            cv2.imshow('image', frame)
        elif s == 4:
            ret, frame = video_capture.read()
            triangles = find_triangles(frame)
            cv2.drawContours( frame, triangles, -1, (0, 255, 0), 3 )
            cv2.imshow('image', frame)
                
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
    
def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares
#Not Working currently
def find_triangles(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    triangles = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 3 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in xrange(3)])
                    if max_cos < 0.1:
                        triangles.append(cnt)
    return triangles    
    
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def cascadeUS(frame, c1, c2, c3, c4, c5, c6, c7):
	global language
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
		minNeighbors=5,
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
	 
	signalAhead = signalAheadCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c7),
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
	
if __name__ == "__main__":
   main()
   sys.exit()