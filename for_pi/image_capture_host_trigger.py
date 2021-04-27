# Runs a flask web app that allows pictures to be taken on trigger click
from flask import Flask, request, render_template
from picamera import PiCamera
from datetime import datetime

app = Flask(__name__)


@app.route('/image_capture', methods=['POST', 'GET'])
def image_capture():
    try:
        take_picture = request.form.get("take_picture")
        print('Take picture? = ', take_picture)
        if take_picture == "t":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_src = "static/raw-image" + timestamp + ".png"
            camera.start_preview()
            # sleep(2)
            camera.capture(image_src)
            camera.stop_preview()
            return render_template('image_capture.html')
    finally:
        return render_template('image_capture.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')