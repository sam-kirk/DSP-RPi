#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:55:22 2020

@author: sam
"""
from flask import Flask, session, request, render_template, redirect, url_for, send_from_directory, send_file
import os
from time import sleep
from datetime import datetime
import glob
from image_class import Image
import itertools
import re

#if running on pi image_capture can run
try:
    from picamera import PiCamera
    is_pi = True
except ImportError:
    is_pi = False

# Plain Old Python defs
# takes all images in file path and creates a new Image object for each
# Image objects are appended to a list and the list is returned
'''def load_image_set(f_path, term):
    print("---- Start")
    images = []  # for storing image objects for this set
    for file in glob.glob(f_path + term):  # for each file that matches the term in the given filepath
        # only process the raw images
        if len(re.findall("_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\Z", file.split(".")[0])) > 0:
            image = Image(file.split("/")[-1], f_path)  # get the filename at the end of the glob path
            images.append(image)

    return images'''


def mock_function(images):  # todo rename
    print("---- Mock Function Called")
    #images[0].preprocess_image()
    #images[0].create_ndvi_images(True)
    #images[0].create_colour_bar_image()
    #images[0].create_cmap_image()
    #images[0].object_detection()
    images[0].main_crop_extraction()
    #images[0].is_match('working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)

    #images[0].process_image_full(True, 'working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)
    '''for image in images:
        image.process_image_full(True, 'working_image_sets/raw_image_blue_filter copy/2021-03-25_14-07-57.png', 10)'''


app = Flask(__name__)

# should be secret but not required for local hosting
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route("/home")
def home():
    print("-home")
    return render_template("home.html")


# controller function
@app.route("/analysis_action", methods=["POST"])
def analysis_action():
    print("-analysis_action")
    fpath = request.form.get("fpath")
    if fpath == "":
        success = False
        fpath = "[Empty]"
    else:
        # check file exists
        # if file does not exist
            # success = False
            # fpath = "[Empty]"
        # else: (all good)
        # split the filepath into name and path
        fname = fpath.rsplit("/", 1)[-1]
        fpath = fpath.rsplit("/", 1)[0] + "/"
        image = Image(fname, fpath)
        paths = image.process_image(True)  # run analysis with normalisation

        # take returned values and make them a flat list
        tple = paths[1]
        paths.pop(1)
        paths.insert(1, tple[0])
        paths.insert(2, tple[1])

        titles = ["Original Image", "Pre Processed Image", "NDVI Grey", "NDVI Colour", 
                  "NDVI Colour Bar", "NDVI Colour Maps", "Object Detection", "Crop Extraction"]

        image_HTML = "<div class='output' style='padding: 10px 0px'><h3>Original Image</h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;'><img class='output_image' src='static/images/2021-03-25_14-08-07.png'><p>static/images2021-03-25_14-08-07.png</p></div>"
        for i in range(len(paths)):
            new_section = "<div class='output' style='padding: 10px 0px'><h3>" + titles[i+1] + "</h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;'><img class='output_image' src='" + paths[i] + "'><p>" + paths[i] + "</p></div>"
            image_HTML = image_HTML + new_section

        session['image_HTML'] = image_HTML
    return redirect("analysis")


@app.route("/analysis", methods=["POST", "GET"])
def analysis():
    print("-analysis")
    try:
        image_HTML = session['image_HTML']
    except:
        image_HTML = '<p>Nada</p>'

    return render_template("analysis.html", image_HTML=image_HTML)


@app.route("/image_match")
def image_match():
    print("-image_match")
    return render_template("image_match.html")


# @app.route("/analysis", methods=["POST", "GET"])
@app.route("/help", methods=["POST", "GET"])
def help():
    print("-help")
    return render_template("help.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    #directory = "working_image_sets/raw_image_blue_filter/"
    #name_term = "*.png"  # can be changed for different patterns or filetypes
    #image_list = load_image_set(directory, name_term)
    #mock_function(image_list)


