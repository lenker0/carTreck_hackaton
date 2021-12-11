from flask import Flask, render_template
# from imutils.video import VideoStream
import time
from flask import Response
import imutils
from liveness_demo import LivenessDetector

app = Flask(__name__)

# vs = VideoStream(src=0).start()
# time.sleep(2.0)
global ld 
ld = LivenessDetector()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/my-link/')
def my_link():
	(output, frame) = ld.detect()
	return render_template('index.html', output = output, frame = frame)

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
  app.run(debug=True)
