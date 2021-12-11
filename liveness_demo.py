from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

class LivenessDetector:
	def __init__(self):

		# load our serialized face detector from disk
		print("[INFO] loading face detector...")
		protoPath = os.path.sep.join(["./face_detector", "deploy.prototxt"])
		modelPath = os.path.sep.join(["./face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
		self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# load the liveness detector model and label encoder from disk
		print("[INFO] loading liveness detector...")
		self.model = load_model("liveness.model")
		self.le = pickle.loads(open("le.pickle", "rb").read())

	def detect(self):
		
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
		flag = 0

		while abs(flag) != 3:
			frame = vs.read()
			frame = imutils.resize(frame, height=480, width=640)

			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
				(300, 300), (104.0, 177.0, 123.0))

			self.net.setInput(blob)
			detections = self.net.forward()

			for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]

				if confidence > 0.5:

					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					startX = max(0, startX)
					startY = max(0, startY)
					endX = min(w, endX)
					endY = min(h, endY)

					face = frame[startY:endY, startX:endX]
					face = cv2.resize(face, (32, 32))
					face = face.astype("float") / 255.0
					face = img_to_array(face)
					face = np.expand_dims(face, axis=0)

					preds = self.model.predict(face)[0]
					j = np.argmax(preds)
					label = self.le.classes_[j]

					if preds[j] > 0.50 and j == 1:
						flag += 1
					else:
						flag -= 1

					#cv2.imshow("Frame", frame)

		if flag == 3:
			ans = 'Real with {:.4f} accuracy'.format(preds[j])
			
		if flag == -3:
			ans = 'Fake with {:.4f} accuracy'.format(preds[j])

		frame_path = "./static/test.png"
		cv2.imwrite(frame_path, frame)
		vs.stop()
		return (ans,"test.png")