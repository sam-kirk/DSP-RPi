# timed image capture
from picamera import PiCamera
from time import sleep

if __name__ == '__main__':
    cont = True
    print('-- TIMED IMAGE CAPTURE --')
    while cont:
        t = input('How many seconds delay? (minimum 5s)')
        if t < 5:
            t = 5
        m = input('How long is your flight time in minutes?')
        x = m // t * 60

        print('Every' + t + 'seconds a picture will be taken for ' + m + ' minutes.')
        print('This will result in ' + x + 'pictures')

        start = ''
        while start != 'Y' or 'N':
            start = input('Enter y to begin or n to cancel.').upper()

        if start == 'Y':
            for i in range(x):
                sleep(t)
                camera.capture('/home/pi/Desktop/image%s.jpg' % i)

        resp = input('Exit? (y/n)').upper()
        if resp == 'Y':
            cont = False
