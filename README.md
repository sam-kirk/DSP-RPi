<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Raspberry Pi NoIR as a Low-cost NDVI Remote Sensing Device for Land Monitoring</h3>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This system was developed to allow more people access to NDVI remote sensing technology to assess plant health with the goal of contibuting to Smart Farming research.

It uses a RPi NoIR camera to take NIR birds eye view images of crops. The images are then processed and analysed for insights which are presented in a local web-app front-end.


### Built With

* []()Python
* []()OpenCV
* []()Flask



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
2. Install the dependancies (all imports listed below)
   ```sh
   from flask import Flask, session, request, render_template, redirect
   import glob
   import re
   import cv2
   import numpy as np
   import matplotlib.pyplot as plt
   from imutils import contours
   from skimage import measure
   import imutils
   import matplotlib
   ```
   
   On RPi
   ```sh
   from flask import Flask, request, render_template
   from picamera import PiCamera
   from datetime import datetime
   from time import sleep
   ```


<!-- USAGE EXAMPLES -->
## Usage
To take images with RPi:
1.Use the for_pi/image_capture_timed.py or for_pi/image_capture_host_trigger.py

  a.For the timed trigger:
    1. Run the script on the RPi and follow the console prompts
    2. It will set the capture to go off at timed intervals for the amount of time you set

  b.For the host trigger scripts:
    1. Run the script on the RPi making sure to host the server on your local network
    2. Using another device navigate to the page and tap the "take picture" button to trigger an image capture on the RPi.

2. Once the picture have been taken use a USB stick, or any other method, to transfer the files from the RPi to your main computer
3. The images should be placed in the web_host/static/image folder
4. Now run web_host/main_host.py.
5. Next, navigate to the local webpage indicated in the console.
6. Finally, the webpage contains instructions on how to analyse the images and inerpret the results.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
