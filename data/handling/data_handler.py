import random
import math
import os
import csv
import numpy as np
import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from data.handling.slice_record import *

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
    def __init__(self, data_path, index_path):
        self.data_path = data_path

        self.records = []
        with open(index_path + '.csv', 'rt') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.records.append(slice_record_from_row(row))

        self.abnormal = [r for r in self.records if r.is_abnormal]
        self.healthy = [r for r in self.records if not r.is_abnormal]

        self.print_stats()

    def load_and_preprocess_image_set(self, index):
        images = np.zeros((len(index), 256, 256))
        for i, record in enumerate(index):
            images[i] = pre_process(record.load_slice_image(self.data_path))
        return images

    # Loads image data
    def load_images(self):
        # [Healthy, Unhealthy]
        healthy_labels = np.tile([1, 0], (len(self.healthy), 1))
        abnormal_labels = np.tile([0, 1], (len(self.abnormal), 1))

        print('Loading abnormal slice images...')
        abnormal = self.load_and_preprocess_image_set(self.abnormal)

        print('Loading healthy slice images...')
        healthy = self.load_and_preprocess_image_set(self.healthy)

        return healthy, healthy_labels, abnormal, abnormal_labels

    def train_test_split(self, data, labels, portion_test):
        n_test = math.floor(len(data) * portion_test)
        test = random.sample(range(len(data)), n_test)
        train = [i for i in range(len(data)) if i not in test]

        return data[train], labels[train], data[test], labels[test]

    # h: healthy, a: abnormal, tr: train, te: test
    def load_dataset(self, portion_test=0.1):
        h, h_labels, abnormal, a_labels = self.load_images()

        h_tr, h_labels_tr, h_te, h_labels_te = self.train_test_split(h, h_labels, portion_test)
        a_tr, a_labels_tr, a_te, a_labels_te = self.train_test_split(abnormal, a_labels, portion_test)

        train_features = np.concatenate((h_tr, a_tr))
        train_labels = np.concatenate((h_labels_tr, a_labels_tr))

        test_features = np.concatenate((h_te, a_te))
        test_labels = np.concatenate((h_labels_te, a_labels_te))

        return train_features, train_labels, test_features, test_labels

    def print_stats(self):
        print('Records loaded.')
        print('Healthy slices: ', len(self.healthy))
        print('Polyp slices:   ', len(self.abnormal))
