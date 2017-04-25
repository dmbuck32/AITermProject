import numpy as np
import PIL
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import time
from directkeys import PressKey, W, A, S, D
from mss import mss

cascPath = 'Cascades/merge_cascade_updated.xml'
cascPath2 = 'Cascades/added_lane_cascade_updated.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stop_cascade.xml'
cascPath6 = ''
cascPath7 = 'Cascades/signal_ahead_cascade.xml'
faceCasc = 'haarcascade_frontalface_default.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
signalAheadCascade = cv2.CascadeClassifier(cascPath7)
faceCascade = cv2.CascadeClassifier(faceCasc)


language = 0
mergeText = ["Merge", unicode("マージ","utf-8"), unicode("Unir","utf-8"), unicode("fusionner","utf-8")]
addedLanesText = ["Added Lane", unicode("追加されたレーン","utf-8"), unicode("Carril añadido","utf-8"), unicode("Voies ajoutées","utf-8")]
pedestrianText = ["Pedestrian Crossing", unicode("横断歩道","utf-8"), unicode("cruce peatonal","utf-8"), unicode("passage piéton","utf-8")]
laneEndsText = ["Lane Ends", unicode("レーンエンド","utf-8"), unicode("Carril termina","utf-8"), unicode("La voie se termine","utf-8")]
stopText = ["Stop", unicode("やめる","utf-8"), unicode("Pare","utf-8"), unicode("Arrêtez","utf-8")]
stopAheadText = ["Stop Ahead", unicode("この先、一旦停止","utf-8"), unicode("Pare a continuación","utf-8"), unicode("Arrêt devant","utf-8")]
signalAheadText = ["Signal Ahead", unicode("この先、信号有り","utf-8"), unicode("Señal Adelante","utf-8"), unicode("signal devant","utf-8")]

def convert(in_position):
    in_position += 101
    in_position /= 100.0
    return in_position

def detect_signs(frame, c1, c2, c3):
	gray = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	merge = mergeCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c1),
		minNeighbors=5,
		minSize=(20, 20),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	pedestrians = pedestrianCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c2),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	stop = stopCascade.detectMultiScale(
		gray,
		scaleFactor=convert(c3),
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)


	for (x, y, w, h) in merge:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

	for (x, y, w, h) in stop:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

	return frame

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def pre_process(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    #ROI restriction
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])

    return processed_img

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


def nothing(self):
    pass

def main():
    cv2.namedWindow('window')
    cv2.createTrackbar('Merge','window',50,98,nothing)
    cv2.createTrackbar('Pedestrian','window',50,98,nothing)
    cv2.createTrackbar('Stop','window',50,98,nothing)

    mon = {'top': 40, 'left': 0, 'width': 800, 'height': 600}

    sct = mss()

    while(True):
        c1 = cv2.getTrackbarPos('Merge','window')
        c2 = cv2.getTrackbarPos('Pedestrian','window')
        c3 = cv2.getTrackbarPos('Stop','window')

        #PressKey(W)
        # Grabs an 800 x 600 Window in upper left corner of the screen.
        #screen_grab = np.array(ImageGrab.grab(bbox=(0,40,1000,760)))
        sct.get_pixels(mon)
        screen_grab = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        ##cv2.imshow('window', detect_signs(cv2.cvtColor(np.array(screen_grab), cv2.COLOR_BGR2RGB), c1, c2, c3))
        cv2.imshow('test', detect_signs(cv2.cvtColor(np.array(screen_grab), cv2.COLOR_BGR2RGB), c1, c2, c3))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
