import cv2
import numpy as nmp


class Image:

    filepath = ''
    name = ''
    raw_bitmap = []
    prepro_bitmap = []  # pre-processed bitmap
    ndvi_bitmap = []
    cmap_bitmap = []  # color mapped bitmap

    # constructor only includes name and filepath as all other data is dependent on these values
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath

    # load in the raw image as a bitmap
    def set_raw_bmap(self):
        full_path = self.filepath + self.name
        self.raw_bitmap = cv2.imread(full_path)

    def save_raw_bitmap(self, new_filepath):
        destination = new_filepath + self.name
        cv2.imsave(destination, self.raw_bitmap)


    # pre-process the bitmap for better analysis
    def set_prepro_bitmap(self):
        prepro_bitmap = self.raw_bitmap
        self.prepro_bitmap = prepro_bitmap

    def save_prepro_bitmap(self, new_filepath):
        destination = new_filepath + self.name
        cv2.imsave(destination, self.prepro_bitmap)


    # calculate ndvi reading per pixel
    def set_ndvi_bitmap(self):
        # split image into rgb channels
        b_ch, g_ch, r_ch = cv2.split(self.prepro_bitmap)

        # convert to array of floats for matrix calculations
        r_ch = nmp.asarray(r_ch, dtype=nmp.float32)
        b_ch = nmp.asarray(b_ch, dtype=nmp.float32)

        try:
            # ndvi equation
            ndvi_bitmap = (r_ch - b_ch)/(r_ch + b_ch)
        except ZeroDivisionError:
            print('NDVI Base divide by zero')

        self.ndvi_bitmap = ndvi_bitmap.tolist()

    def save_ndvi_bitmap(self, new_filepath):
        destination = new_filepath + self.name
        cv2.imsave(destination, self.ndvi_bitmap)


    # apply the colourmap to ndvi image for better readability
    def set_cmap_bitmap(self):
        cmap_bitmap = self.ndvi_bitmap
        self.cmap_bitmap = cmap_bitmap

    def save_cmap_bitmap(self, new_filepath):
        destination = new_filepath + self.name
        cv2.imsave(destination, self.cmap_bitmap)

