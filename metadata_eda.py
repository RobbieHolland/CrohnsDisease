import random
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

from data.handling.parse_labels import DataParser

class MetadataEDA:
    def __init__(self, label_path, data_path, out_path):
        self.out_path = out_path
        self.reader = DataParser(label_path, data_path)
        self.data_path = data_path
        self.metadata = self.reader.shuffle_read()

        self.metadata.print_statistics()

    def centroid_histogram(self):
        centroids = self.metadata.centroid_histogram()
        plt.hist(centroids, normed=True, bins=30)
        plt.ylabel('Centroid height distribution')
        plt.show()

eda = MetadataEDA('./data/cases/', '/vol/bitbucket/rh2515/CT_Colonography', 'data/tfrecords')
eda.centroid_histogram()
