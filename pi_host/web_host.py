#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:55:22 2020

@author: sam
"""
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from picamera import PiCamera
from time import sleep
from datetime import datetime
import glob

app = Flask(__name__)

def get_file_names():
    jpg_files = []
    for file in glob.glob("static/*.jpg"):
        jpg_files.append(file)
    return jpg_files

@app.route('/')
def home():
    imgs = get_file_names()
    return render_template('home.html', imgs = imgs)

@app.route('/capture')
def capture():
    timestamp = datetime.now().strftime("%d-%b-%Y[%H:%M:%S]")
    image_src = "static/image" + timestamp + ".jpg"
    print('bing')
    camera.start_preview()
    #sleep(2)
    camera.capture(image_src)
    camera.stop_preview()
    print('bong')
    return redirect(url_for('home'))

if __name__ == '__main__':
    camera = PiCamera()
    app.run(host= '0.0.0.0')