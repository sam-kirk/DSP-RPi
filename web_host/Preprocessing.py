import cv2
from datetime import datetime
import glob
import numpy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

C_DATE = datetime.now().strftime("%Y-%m-%d")

#function takes raw images and saves a new ndvi image in out_image dir
def ndvi(img):
    #split image into rgb channels
    b_ch, g_ch, r_ch = cv2.split(img[1])
    
    #convert to array of floats for matrix calculations
    r_ch = numpy.asarray(r_ch, dtype=numpy.float32)
    b_ch = numpy.asarray(b_ch, dtype=numpy.float32)
    
    #create ndvi calculation components
    rb_diff = (r_ch - b_ch)
    rb_sum = (r_ch + b_ch)

    #redBlueSum[redBlueSum ==0] = 0.01
    
    #calculate ndvi
    ndvi_i = rb_diff/rb_sum

    plt.imsave('out_images/' + C_DATE + '/' + img[0] + '.png', ndvi_i, vmin=-1.0, vmax=1.0, format='png')
    
    # for color mapping
    # fastiecm=LinearSegmentedColormap.from_list('mylist', colors)
    # plt.imsave(imageOutPath + processedImgFilename +'.png',arrNDVI,cmap=fastiecm, vmin=-1.0, vmax=1.0, format='png')


# function searches in raw images dir for all png files and returns them as a list
def get_images():
    imgs = []
    f_path = "static/raw_images/" + C_DATE + "/*.png"
    for file in glob.glob(f_path):
        img_pair = [file.split('/')[-1].split('.')[0], cv2.imread(file)]
        imgs.append(img_pair)
    return imgs


if __name__ == '__main__':
    # get image and names
    print(C_DATE)
    print(cv2.version)

    imgs = get_images()
    print(imgs)

    #process images
    for i in range(len(imgs)):
        #print('new images \n',imgs[i])
        ndvi(imgs[i])