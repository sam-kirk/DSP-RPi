# timed image capture to run on Pi
from picamera import PiCamera
from time import sleep
from datetime import datetime

if __name__ == '__main__':
    camera = PiCamera()
    
    cont = True    
    print('-- TIMED IMAGE CAPTURE --')

    # continue until user exits
    while cont:
        t = float(input('How long is your flight time in minutes?\n'))
        d = float(input('How many seconds delay? (minimum 3s)\n'))
        # 5s minimum to allow time for capture and save
        if d < 3:
            d = 3
        
        # how many pictures can be taken in that time with that delay
        x = int(t * 60//d)
        print(x)
        print('Every ' + str(d) + ' seconds a picture will be taken for ' + str(t) + ' minutes.')
        print('This will result in ' + str(x) + ' pictures')

        # start of image capture loop
        for i in range(x):
            print('image ' + str(i+1) + ' start')
            sleep(d)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_src = "raw_images/" + timestamp + ".png"
            camera.capture(image_src)
            print('image ' + str(i) + ' done')

        # continue or exit
        resp = input('Exit? (y/n)').upper()
        if resp == 'Y':
            cont = False
