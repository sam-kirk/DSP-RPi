# Runs a flask web app that allows pictures to be taken on trigger click
from flask import Flask, request, render_template, redirect
from picamera import PiCamera
from datetime import datetime

app = Flask(__name__)

@app.route('/image-capture', methods=['POST', 'GET'])
def image_capture():
    try:
        take_picture = request.form.get("take_picture")
        print('Take picture? = ', take_picture)
        if take_picture == "t":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_src = "raw_images/" + timestamp + ".png"
            # print('bing')
            camera.start_preview()
            # sleep(2)
            camera.capture(image_src)
            camera.stop_preview()
            # print('bong')
            return render_template('image-capture.html')
    finally:
        return render_template('image-capture.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')