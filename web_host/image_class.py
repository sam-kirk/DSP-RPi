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
        r_ch = np.asarray(r_ch, dtype=np.float32)
        b_ch = np.asarray(b_ch, dtype=np.float32)

        try:
            # ndvi equation
            ndvi_bitmap = (r_ch - b_ch)/(r_ch + b_ch)
            print(ndvi_bitmap)
            print(ndvi_bitmap.shape)
            ndvi_bitmap = -ndvi_bitmap
            print(np.amin(ndvi_bitmap))
            print(ndvi_bitmap)
        except ZeroDivisionError:
            print("NDVI Base - Divide by zero")

        self.ndvi_bitmap = ndvi_bitmap.tolist()


    def create_ndvi_bitmap2(self):
        "This function performs the NDVI calculation and returns an RGB frame)"
        lowerLimit = 1  # this is to avoid divide by zero and other weird stuff when color is near black
        original = self.prepro_bitmap
        # First, make containers
        oldHeight, oldWidth = original[:, :, 0].shape
        ndviImage = np.zeros((oldHeight, oldWidth, 3), np.uint8)  # make a blank RGB image

        # Now get the specific channels. Remember: (B , G , R)
        red = original[:, :, 2].astype('float')
        blue = original[:, :, 0].astype('float')

        # Perform NDVI calculation
        summ = red + blue
        summ[summ < lowerLimit] = lowerLimit  # do some saturation to prevent low intensity noise

        ndvi = (((red - blue) / (summ) + 1) * 127).astype('uint8')  # the index

        redSat = (ndvi - 128) * 2  # red channel
        bluSat = ((255 - ndvi) - 128) * 2  # blue channel
        redSat[ndvi < 128] = 0  # if the NDVI is negative, no red info
        bluSat[ndvi >= 128] = 0  # if the NDVI is positive, no blue info

        # And finally output the image. Remember: (B , G , R)
        # Red Channel
        ndviImage[:, :, 2] = redSat

        # Blue Channel
        ndviImage[:, :, 0] = bluSat

        # Green Channel
        ndviImage[:, :, 1] = 255 - (bluSat + redSat)

        self.ndvi_bitmap = ndviImage #ndvi_bitmap.tolist()


    def save_ndvi_bitmap(self, new_filepath):
        destination = new_filepath
        plt.imsave(destination, self.ndvi_bitmap)


    # apply the colourmap to ndvi image for better readability
    def create_cmap_bitmap(self):
        cmap_bitmap = self.ndvi_bitmap
        print(cmap_bitmap)
        self.cmap_bitmap = cmap_bitmap

    def save_cmap_bitmap(self, new_filepath):
        destination = new_filepath
        plt.imsave(destination, self.cmap_bitmap)


    # function takes a raw image and populates all the values and saves them in the directory
    def process_image_full(self):

        save_path = self.filepath + self.name.split(".")[0]

        print("nope")
        self.set_raw_bmap()

        self.create_prepro_bitmap()
        self.save_prepro_bitmap(save_path + "_prepro.png")

        self.create_ndvi_bitmap()
        self.save_ndvi_bitmap(save_path + "_ndvi.png")

        self.create_cmap_bitmap()
        self.save_cmap_bitmap(save_path + "_cmap.png")


    @staticmethod
    def test():
        print("beepboop")
        return True

