import cv2


class Image:

    filepath = ''
    name = ''
    raw_bitmap = []
    prepro_bitmap = []  # pre-processed bitmap
    ndvi_bitmap = []
    cmap_bitmap = []  # color mapped bitmap

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath

    def set_raw_bmap(self):
        full_path = self.filepath + self.name
        self.raw_bitmap = cv2.imread(full_path)

    def set_prepro_bitmap(self):
        self.prepro_bitmap = preprocess(self.raw_bitmap)

    def set_ndvi_bitmap(self):
        self.ndvi_bitmap = ndvi(self.prepro_bitmap)

    def set_cmap_bitmap(self):
        self.cmap_bitmap = cmap(self.ndvi_bitmap)