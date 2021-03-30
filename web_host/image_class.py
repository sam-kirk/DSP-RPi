import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

    #self.save_prepro_bitmap(save_path + "_prepro")
    def save_prepro_bitmap(self, new_filepath):
        destination = new_filepath
        cv2.imwrite(destination, self.prepro_bitmap)


    # calculate ndvi reading per pixel
    def create_ndvi_bitmap(self):
        # split image into rgb channels
        b_ch, g_ch, r_ch = cv2.split(self.prepro_bitmap)

        # convert to array of floats for matrix calculations
        r_ch = np.asarray(r_ch).astype('float')
        b_ch = np.asarray(b_ch).astype('float')

        r_b_sum = (r_ch+b_ch)
        r_b_sum[r_b_sum == 0] = 0.01  # ensures no divide by zero errors and reduces noise
        # ndvi equation
        ndvi_bitmap = (r_ch - b_ch) / r_b_sum
        self.ndvi_bitmap = ndvi_bitmap.tolist()


    def save_ndvi_bitmap(self, new_filepath):
        destination = new_filepath
        plt.imsave(destination, self.ndvi_bitmap, cmap="binary", vmin=0.0, vmax=np.amax(self.ndvi_bitmap))


    # apply the colourmap to ndvi image for better readability
    def create_cmap_bitmap(self):
        cmap_bitmap = self.ndvi_bitmap
        self.cmap_bitmap = cmap_bitmap

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


    def analyse_image(self):
        # Based on code by Adrian Rosebrock 2016 accessed March 2021 at
        # https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
        from imutils import contours
        from skimage import measure
        import imutils

        # construct the argument parse and parse the arguments
        image = cv2.imread(self.filepath + self.name.split(".")[0] + "_ndvi.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # threshold the image to reveal light regions in the
        # blurred image
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(thresh, connectivity=1, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 300:
                mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to
        # right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            print('beep beep')
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(cX), int(cY)), int(radius),
                       (0, 0, 255), 3)
            cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("beep", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def test():
        print("beepboop")
        return True

