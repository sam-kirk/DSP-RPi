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
app.secret_key = b'_5#y2L"F4Q9z\n\xec]/'


@app.route("/home")
def home():
    print("-home")
    return render_template("home.html")


# controller function
@app.route("/analysis_action", methods=["POST"])
def analysis_action():
    print("-analysis_action")
    fpath = request.form.get("option")
    if fpath == "":
        fpath = "[Empty]"
    else:
        # else: (all good)
        # split the filepath into name and path
        fname = fpath.rsplit("/", 1)[-1]
        fpath = fpath.rsplit("/", 1)[0] + "/"
        image = Image(fname, fpath)
        paths = image.process_image(True)  # run analysis with normalisation

        # take returned values and make them a flat list
        tple = paths[2]
        paths.pop(2)
        paths.insert(2, tple[0])
        paths.insert(3, tple[1])

        titles = ["Original Image", "Pre Processed Image", "NDVI Grey", "NDVI Colour",
                  "NDVI Colour Bar", "NDVI Colour Map", "Object Detection", "Crop Extraction"]

        image_HTML = ""
        for i in range(len(paths)):
            new_section = "<div class='output' style='padding: 10px 0px'><h3>" + titles[i] +\
                          "</h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;'><img class='output_image' src='" +\
                          paths[i] + "'><p>@location " + paths[i] + "</p></div>"
            image_HTML = image_HTML + new_section

        session['image_HTML'] = image_HTML
    return redirect("analysis")

# controller function
@app.route("/image_match_action", methods=["POST"])
def image_match_action():
    print("-image_match_action")
    fpath1 = request.form.get("option1")
    fpath2 = request.form.get("option2")

    session['match_HTML'] = ""

    if fpath1 == "" or fpath2 == "":
        fpath1 = "[Empty]"
        fpath2 = "[Empty]"
    else:
        # split the filepath into name and path
        fname1 = fpath1.rsplit("/", 1)[-1]
        fpath1 = fpath1.rsplit("/", 1)[0] + "/"
        image = Image(fname1, fpath1)

        paths = []
        print(image.full_path)
        print("fpath op 2 = ", fpath2)
        res = image.is_match(fpath2, 10)  # run match with 10 good matches

        paths.append(res[1])
        titles = ["Image Match"]
        print('p= ', paths)
        match_HTML = ""
        for i in range(len(paths)):
            print(paths[i], "slut")
            new_section = "<div class='output' style='padding: 10px 0px'><h3>" + titles[i] + ' - <i>' + str(res[0]) +\
                          "</i></h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;'><img class='output_image' src='" +\
                          paths[i] + "'><p>@location " + paths[i] + "</p></div>"
            match_HTML = match_HTML + new_section
        session['match_HTML'] = match_HTML
    return redirect("image_match")


@app.route("/analysis", methods=["POST", "GET"])
def analysis():
    print("-analysis")
    f_path = 'static/images/'
    term = '*.png'
    raw_files = []  # for storing image objects for this set
    for file in glob.glob(f_path + term):  # for each file that matches the term in the given filepath
        # only process the raw images
        if len(re.findall("_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\Z", file.split(".")[0])) > 0:
            raw_files.append(file)
    print(raw_files)
    try:
        image_HTML = session['image_HTML']
    except:
        image_HTML = '<p>Choose an image to analyse and the output will be shown below</p>'

    return render_template("analysis.html", image_HTML=image_HTML, raw_files=raw_files)


@app.route("/image_match", methods=["POST", "GET"])
def image_match():
    print("-image_match")
    f_path = 'static/images/'
    term = '*.png'
    raw_files = []  # for storing image objects for this set
    for file in glob.glob(f_path + term):  # for each file that matches the term in the given filepath
        # only process the raw images
        if len(re.findall("_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\Z", file.split(".")[0])) > 0:
            raw_files.append(file)
    try:
        match_HTML = session['match_HTML']
    except:
        match_HTML = '<p>Choose two images to compare and the output will be shown below</p>'

    return render_template("image_match.html", match_HTML=match_HTML, raw_files=raw_files)


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


