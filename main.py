import cv2
import glob
import numpy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


#function takes raw images and saves a new ndvi image in out_image dir
def ndvi(img,img_nm):
    #split image into rgb channels
    b_ch, g_ch, r_ch = cv2.split(img)
    
    #convert to array of floats for matrix calculations
    r_ch = numpy.asarray(r_ch, dtype=numpy.float32)
    b_ch = numpy.asarray(b_ch, dtype=numpy.float32)
    
    #create ndvi calculation components
    rb_diff = (r_ch - b_ch)
    rb_sum = (r_ch + b_ch)
    
    #calculate ndvi
    ndvi_l = rb_diff/rb_sum
    
    print(ndvi_l)
    plt.imsave('out_images/' + img_nm + '.png', ndvi_l, vmin=-1.0,vmax=1.0)


#function searches in raw images dir for all jpg files and returns them as an array
def get_images():
    imgs = []
    for file in glob.glob("raw_images/*.jpg"):
        imgs.append(cv2.imread(file))
    return imgs


#function searches in raw images dir for all jpg files and returns their names
def get_image_names():
    img_nms = []
    for file in glob.glob("raw_images/*.jpg"):
        file = file.split('/')[1]
        img_nms.append(file.split('.')[0])
    return img_nms


#main driver
if __name__ == '__main__':
    #get image and names
    imgs = get_images()
    img_nms = get_image_names()
    #print(imgs)
    
    #process images
    for i in range(len(img_nms)):
        #print('new images \n',imgs[i])
        ndvi(imgs[i], img_nms[i])