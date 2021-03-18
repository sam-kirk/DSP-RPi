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
        print('Navigate yo home')
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')