#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:55:22 2020

@author: sam
"""
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
import os
from time import sleep
from datetime import datetime
import glob

#if running on pi image_capture can run
try:
    from picamera import PiCamera
    is_pi = True
except ImportError:
    is_pi = False

# Plain Python defs

app = Flask(__name__)


@app.route('/home')
def home():
    if not is_pi:
        # todo make if not pi don't show take picture sidebar option
        print('beep')
    return render_template('home.html')


@app.route('/image-capture', methods=['POST', 'GET'])
def image_capture():
    is_pi = True
    if is_pi:
        try:
            take_picture = request.form.get("take_picture")
            if take_picture == "t":
                timestamp = datetime.now().strftime("-%Y-%m-%d[%H:%M:%S]")
                image_src = "static/image" + timestamp + ".png"
                print('bing')
                camera.start_preview()
                # sleep(2)
                camera.capture(image_src)
                camera.stop_preview()
                print('bong')
                return render_template('image-capture.html')
        finally:
            return render_template('image-capture.html')
    else:
        return redirect('/home')


if __name__ == '__main__':

    app.run(host='0.0.0.0')