#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:55:22 2020

@author: sam kirk

@purpose: This runs the web host for the NDVI analysis system navigate to 0.0.0.0/home on run
"""
from flask import Flask, session, request, render_template, redirect, url_for, send_from_directory, send_file
import os
from time import sleep
from datetime import datetime
import glob
from image_class import Image
import itertools
import re

app = Flask(__name__)

# should be secret but not required for local hosting
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


# find all the raw files in static/images
def find_files():
    f_path = 'static/images/'
    term = '*.png'
    raw_files = []  # for storing image objects for this set
    for file in glob.glob(f_path + term):  # for each file that matches the term in the given filepath
        # only allow the raw images i.e. filenames without a text tag
        if len(re.findall("_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\Z", file.split(".")[0])) > 0:
            raw_files.append(file)
    print(raw_files)
    return raw_files


# disable caching to allow for new image matches to be completed
@app.after_request
def add_header(r):
    """
    Code fix from stackoverflow: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/analysis_action", methods=["POST"])
def analysis_action():
    print("-analysis_action")
    fpath = request.form.get("option")
    if fpath == "":
        fpath = "[Empty]"
    else:
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
        print('p= ', paths)
        image_HTML = ""
        for i in range(len(paths)):
            new_section = "<div class='output' style='padding: 10px 0px'><h3>" + titles[i] +\
                          "</h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;'>" \
                          "<img class='output_image' src='" + paths[i] + "'><p>@location " + paths[i] + "</p></div>"
            image_HTML = image_HTML + new_section

        session['image_HTML'] = image_HTML
    return redirect("analysis")


@app.route("/image_match_action", methods=["POST"])
def image_match_action():
    print("-image_match_action")
    fpath1 = request.form.get("option1")
    fpath2 = request.form.get("option2")
    if fpath1 == "" or fpath1 == "":
        fpath1 = "[Empty]"
        fpath2 = "[Empty]"
    else:
        # split the filepath into name and path
        fname1 = fpath1.rsplit("/", 1)[-1]
        fpath1 = fpath1.rsplit("/", 1)[0] + "/"
        image = Image(fname1, fpath1)

        paths = []
        res = image.is_match(fpath2, 10)  # run match with 10 good matches
        paths.append(res[1])
        titles = ["Image Match"]
        print('p= ', paths)
        match_HTML = ""
        for i in range(len(paths)):
            new_section = "<div class='output' style='padding: 10px 0px'><h3>" + titles[i] + ' - <i>' + str(res[0]) +\
                          "</i></h3><hr style='border-top: 1px dashed #333333; width: 40%; margin: 0px;" \
                          "'><img class='output_image' src='" + paths[i] + "'><p>@location " + paths[i] + "</p></div>"
            match_HTML = match_HTML + new_section

        session['match_HTML'] = match_HTML
    return redirect("image_match")


@app.route("/analysis")
def analysis():
    print("-analysis")
    drop_down = find_files()
    try:
        image_HTML = session['image_HTML']
    except:
        image_HTML = '<p>Choose an image to analyse and the output will be shown below</p>'

    return render_template("analysis.html", image_HTML=image_HTML, raw_files=drop_down)


@app.route("/image_match")
def image_match():
    print("-image_match")
    drop_down = find_files()
    try:
        match_HTML = session['match_HTML']
    except:
        match_HTML = '<p>Choose two images to compare and the output will be shown below</p>'

    return render_template("image_match.html", match_HTML=match_HTML, raw_files=drop_down)


@app.route("/home")
def home():
    print("-home")
    return render_template("home.html")


@app.route("/help", methods=["POST", "GET"])
def help():
    print("-help")
    return render_template("help.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")


