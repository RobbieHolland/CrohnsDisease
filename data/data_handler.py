import random
import numpy as np
import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import matplotlib.pyplot as plt

from data.parse_labels import DataParser
random.seed(1234)

# Inefficient, have to convert to ITK image and then back
# TODO: Redesign pre-processing
def pre_process(image):
    # Normalise image
    image = whitening(image)

    # Rescale
    itk_image = sitk.GetImageFromArray(np.array(image))
    resample = sitk.ResampleImageFilter()
    scale = sitk.ScaleTransform(2, (2, 2))
    resample.SetTransform(scale)
    resample.SetSize((256, 256))
    itk_image = resample.Execute(itk_image)

    processed_image = sitk.GetArrayFromImage(itk_image)
    return processed_image

class DataHandler:
    def __init__(self, label_path, data_path):
        self.reader = DataParser(label_path, data_path)
        self.records = self.reader.read()
        self.abnormal = [r for r in self.records if r.is_abnormal]
        self.healthy = [r for r in self.records if not r.is_abnormal]

        self.n_polyp_slices = sum([len(x.polyp_slices) for x in self.records])

        self.print_stats()

    # Loads image data
    def load_images(self):
        healthy = np.zeros((self.n_polyp_slices, 256, 256))
        abnormal = np.zeros((self.n_polyp_slices, 256, 256))

        healthy_labels = np.zeros(self.n_polyp_slices)
        abnormal_labels = np.ones(self.n_polyp_slices)

        # Load abnormal slices
        print('Loading abnormal slice images...')
        i = 0
        for record in self.abnormal:
            for slice in record.polyp_slices:
                abnormal[i] = pre_process(record.load_slice_image(slice))
                i = i + 1

        # Load random set (size n_polyp_slices) of healthy slices
        print('Loading healthy slice images...')
        random.shuffle(self.healthy)
        for i in range(self.n_polyp_slices):
            healthy_record = self.healthy[i]
            slice = random.randint(0, healthy_record.volume_height - 1)
            healthy[i] = pre_process(healthy_record.load_slice_image(slice))

        return healthy, healthy_labels, abnormal, abnormal_labels

    def load_dataset(self):
        healthy, healthy_labels, abnormal, abnormal_labels = self.load_images()
        features = np.concatenate((healthy, abnormal))
        labels = np.concatenate((healthy_labels, abnormal_labels))

        return features, labels

    def print_stats(self):
        n_studies = len(self.records)
        n_slices = sum([x.volume_height for x in self.records])
        print('Records loaded.')
        print('Studies:        ', n_studies)
        print('Healthy slices: ', n_slices - self.n_polyp_slices)
        print('Polyp slices:   ', self.n_polyp_slices)
