# Sam Kirk
#
# 06/04/2021
#
# UWE DSP Code
# Class for processing NIR images from the PiNoIR module and creating and NDVI image for analysis
#
# With help from:
# - Adrian Rosebrock 2016 at
#   https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
#   accessed March 2021
# - Alexander Mordvintsev & Abid K. 2013 at
#   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
#   accessed March 2021
# - https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656 accessed March 2021

import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import imutils

# constants
FILE_TYPE = ".png"
BLOB_SIZE = 300

# name tails for consistent naming conventions
PREPRO_NAME_TAIL = "_prepro"
NDVI_COLOUR_NAME_TAIL = "_ndvi-c"
NDVI_GRAY_NAME_TAIL = "_ndvi-g"
NDVI_COLOURBAR_NAME_TAIL = "_cbar"
NDVI_COLOUR_MAP_NAME_TAIL = "_cmap"
NDVI_OBJECTS_NAME_TAIL = "_objects"
NDVI_MAIN_CROP_NAME_TAIL = "_mcrop"
NDVI_MATCH_NAME_TAIL = "_match"

# threshold values for classifying areas as dead, unhealthy, healthy and very healthy
# *255 as cv2 is encoded this way
VH = 0.66 * 255
H = 0.33 * 255
UH = 0.22 * 255
D = 0


# model class for image
# contains functions for processing an individual image
class Image:
    name = ""
    dir = ""
    full_path = ""
    raw_bitmap = []

    # constructor only includes name and dir as all other data is dependent on these values
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
        self.full_path = dir + name
        print(dir + name + " object was created")

    # adds a new tail to the end of the image name e.g. foo.png --> foo_tail.png
    # params - tail:string
    # returns - new_name:string
    def attach_name_tail(self, tail):
        new_name = self.full_path.split(".")[0] + tail + FILE_TYPE
        return new_name

    # run all the image processing functions
    # params - norm:boolean, comp_img_path:string, comps:int
    # returns - None
    def process_image_full(self, norm, comp_img_path, comps):
        data = [self.preprocess_image(), self.create_ndvi_images(norm), self.create_colour_bar_image(),
                self.create_cmap_image(), self.object_detection(), self.main_crop_extraction(),
                self.is_match(comp_img_path, comps)]
        return data

    # run all the image processing functions except the comparison
    # params - norm:boolean
    # returns - None
    def process_image(self, norm):
        data = [self.preprocess_image(),
                self.create_ndvi_images(norm),
                self.create_colour_bar_image(),
                self.create_cmap_image(),
                self.object_detection(),
                self.main_crop_extraction()]
        return data

    # reads in original image pre-processes it then saves it with a modified name
    # params - None
    # returns - [file save location]:string
    def preprocess_image(self):
        # read in image
        img = cv2.imread(self.full_path)
        # add preprocessing logic here if needed
        # save as new image
        cv2.imwrite(self.attach_name_tail(PREPRO_NAME_TAIL), img)
        return self.attach_name_tail(PREPRO_NAME_TAIL)

    # calculates the ndvi reading per pixel and saves both a colour and greyscale image
    # the greyscale image can be normalised or not depending on the corresponding parameter 'normalise'
    # params - normalise:boolean
    # returns - [file save location gray]:string, [file save location colour]:string
    def create_ndvi_images(self, normalise):
        img = cv2.imread(self.attach_name_tail(PREPRO_NAME_TAIL))
        # split image into rgb channels
        b_ch, g_ch, r_ch = cv2.split(img)

        # with No IR filter red is considered to be mostly IR
        # as a blue filter was used all visible light is in the blue channel
        # convert to array of floats for matrix calculations
        r_ch = np.asarray(r_ch).astype('float')
        b_ch = np.asarray(b_ch).astype('float')

        # separate sum to allow for limit
        r_b_sum = (r_ch + b_ch)
        r_b_sum[r_b_sum == 0] = 0.01  # ensures no divide by zero errors and reduces noise
        # ndvi equation
        ndvi_bitmap = (r_ch - b_ch) / r_b_sum

        # easiest way to convert to cv2 type is through write then read todo could be inefficient
        # save original ndvi bitmap
        save_path = self.attach_name_tail(NDVI_COLOUR_NAME_TAIL)
        plt.imsave(save_path, ndvi_bitmap)
        # read the bitmap as a cv2 type
        img = cv2.imread(save_path)
        # invert the image for colour correction
        img = cv2.bitwise_not(img)
        # save the cv2 image as colour image
        cv2.imwrite(save_path, img)

        # convert to gray image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalise the image todo fine tune
        if normalise:
            img = cv2.subtract(img, (165), None)
            img = cv2.multiply(img, (10), None)

        # save normalised greyscale image
        cv2.imwrite(self.attach_name_tail(NDVI_GRAY_NAME_TAIL), img)
        return self.attach_name_tail(NDVI_GRAY_NAME_TAIL), save_path

    # uses matplotlib plt to create a new image with a colourbar for easier estimation of crop performance and saves it
    # params - None
    # returns - [file save location]:string
    def create_colour_bar_image(self):  # todo neaten
        img = cv2.imread(self.attach_name_tail(NDVI_COLOUR_NAME_TAIL))
        fig, ax = plt.subplots()
        cax = ax.imshow(img, vmin=0, vmax=255)
        ax.set_title('NDVI Image')
        # Add colourbar
        cbar = fig.colorbar(cax, ticks=[0, 0.33*255, 0.66*255, 255], shrink=0.5)
        cbar.ax.set_yticklabels(['<0', '0.33', '0.66', '1'])
        plt.axis('off')
        plt.savefig(self.dir + self.name.split('.')[0] + '_ndvi-c-bar.png', format='png', dpi=800)
        # plt.show()
        return self.dir + self.name.split('.')[0] + '_ndvi-c-bar.png'

    # apply a threshold colourmap to the ndvi image for quicker identification of crop health and save the  new image
    # params - None
    # returns - [file save location]:string
    def create_cmap_image(self):
        # get an original grayscale image for colouring
        img = cv2.imread(self.attach_name_tail(NDVI_GRAY_NAME_TAIL))
        # get a grayscale for creating masks
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blur to reduce noise and add softness to edges
        img_g_blur = cv2.GaussianBlur(img_g, (35, 35), 0)

        '''# erode and dilate to remove small blobs
        img_g_blur = cv2.erode(img_g_blur, None, iterations=2)
        img_g_blur = cv2.dilate(img_g_blur, None, iterations=4)'''

        # create an iterable list of masks to overlay classified by dead, unhealthy, healthy and very healthy thresholds
        masks = [cv2.inRange(img_g_blur, VH, 255), cv2.inRange(img_g_blur, H, VH), cv2.inRange(img_g_blur, UH, H),
                 cv2.inRange(img_g_blur, D, UH)]

        # create label lists for formatting
        colours = [(0, 255, 0), (0, 255, 180), (0, 0, 255), (50, 50, 50)]
        labels = ["Very Healthy", "Healthy", "Unhealthy", "Inanimate/Dead"]

        # build the overlay that will be put on top of the image
        overlay = np.zeros(img.shape, dtype="uint8")
        for (i, mask) in enumerate(masks):
            # find the contours in the mask
            mask = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = imutils.grab_contours(mask)

            # if no contours continue - the image may not necessarily contain all classifications
            if len(mask) == 0:
                continue

            # sort contours by size so only biggest of colour is labeled
            mask = sorted(contours.sort_contours(mask)[0], key=lambda x: cv2.contourArea(x), reverse=True) # todo sometimes not visable

            # for each contour in the mask add to overlay
            for (j, c) in enumerate(mask):
                # bounding rectangle for text
                (x, y, w, h) = cv2.boundingRect(c)
                # fill contour with colour
                cv2.fillPoly(overlay, pts=[c], color=colours[i])
                if j == 0:
                    cv2.putText(overlay, labels[i], (x + h // 2, y + h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # apply overlay
        output = np.zeros(img.shape, dtype="uint8")
        # higher alpha the more prominent the overlay
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, output)

        cv2.imwrite(self.attach_name_tail(NDVI_COLOUR_MAP_NAME_TAIL), output)
        return self.attach_name_tail(NDVI_COLOUR_MAP_NAME_TAIL)

    # create a mask of crop objects from the greyscale image to identify main crop and anomalies
    # params - None
    # returns - [file save location]:string
    def object_detection(self):  # todo needs softening
        # read in file convert to a cv2 grayscale format
        img = cv2.imread(self.attach_name_tail(NDVI_GRAY_NAME_TAIL))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blurring reduces image noise
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)

        # threshold the image to reveal light regions in the
        # cv2 image type is 8 bit encoding i.e. 0-255
        # if the pixel >= 200 it becomes white if pixel < 200 it is black
        # [0] returns threshold value [1] returns bitmap
        thresh = cv2.threshold(blurred, H, 255, cv2.THRESH_BINARY)[1]

        #self.quick_show(thresh)
        #cv2.imwrite(self.attach_name_tail('_threshmask'), thresh)

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
            if num_pixels > BLOB_SIZE:  # number of pixels to be a large blob
                mask = cv2.add(mask, label_mask)



        # find the contours in the mask, then sort them from left to right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # sort contours by area (largest first)
        cnts = sorted(contours.sort_contours(cnts)[0], key=lambda x: cv2.contourArea(x), reverse=True)

        # loop over the contours
        for (i, c) in enumerate(cnts):
            # take the largest contour by area as main crop
            if i == 0:
                # Bounding rectangle for text placement
                (x, y, w, h) = cv2.boundingRect(c)

                # Convex hull creates a contour from another contour
                # It is used to determine if the crop space is ragged and therefore has a large edge effect
                c_h = cv2.convexHull(c)
                cv2.drawContours(img, [c], 0, (255, 0, 0), 3)
                cv2.drawContours(img, [c_h], 0, (255, 0, 200), 3)
                #self.quick_show(img)

                # calculate the perimeter area ration and represent the difference as a percentage
                perimeter_area_ratio_c = cv2.arcLength(c, True) / cv2.contourArea(c)
                perimeter_area_ratio_c_h = cv2.arcLength(c_h, True) / cv2.contourArea(c_h)
                match = 1-(perimeter_area_ratio_c_h/perimeter_area_ratio_c)

                cv2.putText(img, "#" + str(i+1) + " main crop", (x + w//2, y + h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, "edge effect: " + str(round((1 - match) * 100, 1)) + "%",
                            (x + w // 2, (y + h // 2)+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 200), 2)

            else:
                # draw the bright spot on the image and label as an anomaly
                (x, y, w, h) = cv2.boundingRect(c)
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                cv2.drawContours(img, [c], 0, (0, 0, 200), 3)
                cv2.circle(img, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
                cv2.putText(img, "#{} anomaly".format(i + 1), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #self.quick_show(img)
        cv2.imwrite(self.attach_name_tail(NDVI_OBJECTS_NAME_TAIL), img)
        return self.attach_name_tail(NDVI_OBJECTS_NAME_TAIL)

    # extract only the main crop from the image for detailed analysis
    # params - None
    # returns - [file save location]:string
    def main_crop_extraction(self):
        # read in file convert to a cv2 grayscale format
        image = cv2.imread(self.attach_name_tail(NDVI_GRAY_NAME_TAIL))
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
            if num_pixels > BLOB_SIZE:  # number of pixels to be a large blob
                mask = cv2.add(mask, label_mask)

        # find the contours in the mask, then sort them from left to right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # sort contours by area (largest first)
        cnts = sorted(contours.sort_contours(cnts)[0], key=lambda x: cv2.contourArea(x), reverse=True)
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # take the largest contour by area as main crop
            if i == 0:

                '''mask = np.zeros_like(image)  # Create mask where white is what we want, black otherwise
                cv2.drawContours(mask, [c], 0, 255, -1)  # Draw filled contour in mask
                out = np.zeros_like(image)  # Extract out the object and place into output image
                out[mask == x] = image[mask == x]
                self.quick_show(out)'''

                mask = np.zeros_like(image)  # Create mask where white is what we want, black otherwise
                cv2.fillPoly(mask, [c], [255,255,255])  # Draw filled contour in mask

                #cv2.imwrite(self.attach_name_tail('_maincropextract'), mask)
                sel = mask != 255
                #self.quick_show(mask)

                image[sel] = 0
                #self.quick_show(image)
                # Now crop
                # get bounding rectangle to crop to
                (x, y, w, h) = cv2.boundingRect(c)
                # remove indexes that are outside the region of interest
                roi = image[y:y + h, x:x + w]

                # Show the output image
                # self.quick_show(roi)
        cv2.imwrite(self.attach_name_tail(NDVI_MAIN_CROP_NAME_TAIL), roi)
        return self.attach_name_tail(NDVI_MAIN_CROP_NAME_TAIL)

    # takes the 'self' image and one other (as a parameter) and compares their features
    # if enough features are a good match the images are considered a match
    # params - min_match_count:int
    # returns - match:boolean, [file save location]:string
    def is_match(self, second_img_path, min_match_count):
        img1 = cv2.imread(self.attach_name_tail(NDVI_COLOUR_NAME_TAIL))
        img2 = cv2.imread(second_img_path)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # sift
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # feature matching
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > min_match_count:
            match = True
        else:
            match = False

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)

        #self.quick_show(img3)
        cv2.imwrite(self.attach_name_tail(NDVI_MATCH_NAME_TAIL), img3)
        return match, self.attach_name_tail(NDVI_MATCH_NAME_TAIL)

    @staticmethod
    def quick_show(img):
        cv2.imshow("Img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
