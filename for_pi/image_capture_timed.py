# timed image capture to run on Pi
from picamera import PiCamera
from time import sleep
from datetime import datetime

if __name__ == '__main__':
    cont = True
    print('-- TIMED IMAGE CAPTURE --')

    # continue until user exits
    while cont:
        t = input('How many seconds delay? (minimum 5s)')
        # 5s minimum to allow time for capture and save
        if t < 5:
            t = 5

        m = input('How long is your flight time in minutes?')
        # how many picture can be taken in that time with that delay
        x = m // t * 60
        print('Every' + t + 'seconds a picture will be taken for ' + m + ' minutes.')
        print('This will result in ' + x + 'pictures')

        start = ''
        while start != 'Y' or 'N':
            start = input('Enter y to begin or n to cancel.').upper()

        if start == 'Y':
            # start of image capture loop
            for i in range(x):
                sleep(t)
                timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
                image_src = "static/raw-image" + timestamp + ".png"
                camera.capture('image_src')

        # continue or exit
        resp = input('Exit? (y/n)').upper()
        if resp == 'Y':
            cont = False
