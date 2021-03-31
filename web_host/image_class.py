import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import imutils
from collections import Counter


class Image:
    name = ""
    filepath = ""
    raw_bitmap = []
    prepro_bitmap = []  # pre-processed bitmap
    ndvi_bitmap = []
    cmap_bitmap = []  # color mapped bitmap

    # constructor only includes name and filepath as all other data is dependent on these values
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        print(filepath + name + " object was created")

    # load in the raw image as a bitmap
    def set_raw_bmap(self):
        full_path = self.filepath + self.name
        self.raw_bitmap = cv2.imread(full_path)

    # pre-process the bitmap for better analysis
    def create_prepro_bitmap(self):
        prepro_bitmap = self.raw_bitmap
        self.prepro_bitmap = prepro_bitmap

    def save_prepro_bitmap(self, new_filepath):
        destination = new_filepath
        cv2.imwrite(destination, self.prepro_bitmap)

    # calculate ndvi reading per pixel
    def create_ndvi_bitmap(self):
        # split image into rgb channels
        b_ch, g_ch, r_ch = cv2.split(self.prepro_bitmap)

        # with No IR filter red is considered to be mostly IR
        # ss a blue filter was used all visible light is in the blue channel

        # convert to array of floats for matrix calculations
        r_ch = np.asarray(r_ch).astype('float')
        b_ch = np.asarray(b_ch).astype('float')

        # separate sum to allow for limit
        r_b_sum = (r_ch + b_ch)
        r_b_sum[r_b_sum == 0] = 0.01  # ensures no divide by zero errors and reduces noise

        # ndvi equation
        ndvi_bitmap = (r_ch - b_ch) / r_b_sum
        self.ndvi_bitmap = ndvi_bitmap.tolist()  # todo to list needed?

    def save_ndvi_bitmap(self, new_filepath):
        destination = new_filepath
        # Save as binary cmap for greyscale
        # vmin = 0 as NDVI below 0 suggests area is not biomass (likely water)
        # vmax as max value to get most out of the cmap scale todo may not be good idea for comparisons
        # plt.imsave(destination, self.ndvi_bitmap, cmap="binary", vmin=0.0, vmax=np.amax(self.ndvi_bitmap))
        plt.imsave(destination, self.ndvi_bitmap, cmap="binary", vmin=-1, vmax=1)

    # apply the colourmap to ndvi image for better readability
    def create_cmap_bitmap(self):
        img = cv2.imread(self.filepath + self.name.split(".")[0] + "_ndvi.png")
        img = cv2.normalize(img, normalizedData_l1)
        cv2.imshow('v', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # approximate health based on ndvi value
        vh = 0.66 * 255
        h = 0.33 * 255

        maskv=cv2.inRange(img, (vh,vh,vh), (255,255,255))
        maskh=cv2.inRange(img, (h, h, h), (vh, vh, vh))
        mask=cv2.inRange(img, (0, 0, 0), (h, h, h))


        cv2.imshow('maskv', maskv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('maskh', maskh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_cmap_bitmap(self, new_filepath):
        destination = new_filepath
        plt.imsave(destination, self.cmap_bitmap, cmap="Greens_r")

    # function takes a raw image and populates all the values and saves them in the directory
    def process_image_full(self):

        save_path = self.filepath + self.name.split(".")[0]

        self.set_raw_bmap()

        self.create_prepro_bitmap()
        self.save_prepro_bitmap(save_path + "_prepro.png")

        self.create_ndvi_bitmap()
        self.save_ndvi_bitmap(save_path + "_ndvi.png")

        self.create_cmap_bitmap()
        self.save_cmap_bitmap(save_path + "_cmap.png")

    def object_detection(self):
        # Based on code by Adrian Rosebrock 2016 accessed March 2021 at
        # https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/

        # read in file convert to a cv2 grayscale format
        image = cv2.imread(self.filepath + self.name.split(".")[0] + "_ndvi.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurring reduces image noise
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # threshold the image to reveal light regions in the
        # cv2 image type is 8 bit encoding i.e. 0-255
        # if the pixel >= 200 it becomes white if pixel < 200 it is black
        # [0] returns threshold value [1] returns bitmap
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

        # perform a series of erosions and dilations to remove small blobs from the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # perform a connected component analysis on the thresholded image
        # then initialize a mask to store only the "large" components
        labels = measure.label(thresh, connectivity=2, background=0)  # image with labels
        mask = np.zeros(thresh.shape, dtype="uint8")


        # loop over the unique components
        for label in np.unique(labels):

            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels
            label_mask = np.zeros(thresh.shape, dtype="uint8")
            label_mask[labels == label] = 255
            num_pixels = cv2.countNonZero(label_mask)
            # if the number of pixels in the component is sufficiently large
            # then add it to our mask of "large blobs"
            if num_pixels > 300:  # number of pixels to be a large blob todo parametrise this?
                mask = cv2.add(mask, label_mask)

            #if num_pixels > 300:  # number of pixels to be a large blob todo parametrise this?
                #mask = cv2.add(mask, label_mask)


        # find the contours in the mask, then sort them from left to
        # right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # sort contours by area (largest first)
        cnts = sorted(contours.sort_contours(cnts)[0], key=lambda x: cv2.contourArea(x), reverse=True)
        print('cnts---',cnts)
        # loop over the contours
        # todo how do they look without blur?
        for (i, c) in enumerate(cnts):
            # take the largest contour by area as main crop
            if i == 0:
                cv2.drawContours(image, [c], 0, (255, 0, 0), 3)
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.putText(image, "#{} main crop".format(i + 1), (x + h//2, y + h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                # draw the bright spot on the image
                (x, y, w, h) = cv2.boundingRect(c)
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                cv2.drawContours(image, [c], 0, (0, 0, 200), 3)
                cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
                cv2.putText(image, "#{} anomaly".format(i + 1), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        cv2.imshow("Blur", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Highlighted Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def test():
        print("beepboop")
        return True

