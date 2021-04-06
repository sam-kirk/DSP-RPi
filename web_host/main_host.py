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
from image_class import Image
import re

#if running on pi image_capture can run
try:
    from picamera import PiCamera
    is_pi = True
except ImportError:
    is_pi = False

# Plain Python defs

app = Flask(__name__)


@app.route("/home")
def home():
    if not is_pi:
        print("Navigate yo home")
    return render_template("home.html")


@app.route("/image-capture", methods=["POST", "GET"])
def image_capture():
    is_pi = True
    if is_pi:
        try:
            take_picture = request.form.get("take_picture")
            if take_picture == "t":
                timestamp = datetime.now().strftime("-%Y-%m-%d[%H:%M:%S]")
                image_src = "static/image" + timestamp + ".png"
                print("bing")
                camera.start_preview()
                # sleep(2)
                camera.capture(image_src)
                camera.stop_preview()
                print("bong")
                return render_template("image-capture.html")
        finally:
            return render_template("image-capture.html")
    else:
        return redirect("/home")


# takes all images in file path and creates a new Image object for each
# Image objects are appended to a list and the list is returned
def load_image_set(f_path, term):
    print("---- Start")
    images = []  # for storing image objects for this set
    for file in glob.glob(f_path + term):  # for each file that matches the term in the given filepath
        # only process the raw images
        if len(re.findall("_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\Z", file.split(".")[0])) > 0:
            image = Image(file.split("/")[-1], f_path)  # get the filename at the end of the glob path
            images.append(image)

    return images


def mock_function(images):  # todo rename
    print("---- Mock Function Called")
    #images[0].preprocess_image()
    #images[0].create_ndvi_images(True)
    #images[0].create_colour_bar_image()
    #images[0].create_cmap_image()
    #images[0].object_detection()
    #images[0].main_crop_extraction()
    #images[0].is_match('working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)

    #images[0].process_image_full(True, 'working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)
    for image in images:
        image.process_image_full(True, 'working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)


if __name__ == "__main__":
    #app.run(host="0.0.0.0")
    directory = "working_image_sets/raw_image_blue_filter/"
    name_term = "*.png"  # can be changed for different patterns or filetypes
    image_list = load_image_set(directory, name_term)
    mock_function(image_list)


