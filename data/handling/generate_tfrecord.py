'''
Generates TFRecords of the dataset
'''

import math
import random
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

from data.handling.parse_labels import DataParser

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    # Since this will be used to convert an np.array we don't use []
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def resize(image):
    itk_image = sitk.GetImageFromArray(np.array(image))
    resample = sitk.ResampleImageFilter()
    scale = sitk.ScaleTransform(2, (2, 2))
    resample.SetTransform(scale)
    resample.SetSize((256, 256))
    itk_image = resample.Execute(itk_image)
    scaled_image = sitk.GetArrayFromImage(itk_image)

    return scaled_image

def pre_process(image):
    # Normalise image
    image = whitening(image)
    return image

def volume_index(record, slice):
    return record.volume_height - 1 - slice


class TFRecordGenerator:
    def __init__(self, label_path, data_path, out_path):
        self.reader = DataParser(label_path, data_path)
        self.data_path = data_path
        self.records = self.reader.shuffle_read()

        self.print_dataset_statistics()

        self.train_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, 'train.tfrecords'))
        self.test_writer  = tf.python_io.TFRecordWriter(os.path.join(out_path, 'test.tfrecords'))

    def print_dataset_statistics(self):
        print(len(self.records), 'records')
        n_slices = np.sum([len(r.slices) for r in self.records])
        n_polyp_slices = np.sum([len(r.slices) for r in self.records if r.is_abnormal])
        print(n_polyp_slices, 'slices with polyps')
        print(n_slices - n_polyp_slices, 'slices without polyps')

    def _split(self, train_test_split):
        n_records = len(self.records)
        n_test = math.floor(train_test_split * n_records)
        test_indicies = random.sample(range(n_records), n_test)
        test_records = [self.records[i] for i in test_indicies]
        train_records = [r for r in self.records if r not in test_records]

        return train_records, test_records

    def _generate_tfrecords(self, records, writer):
        for record in records:
            print('Loading', record.patient_no)
            data = sitk.ReadImage(record.form_path(self.data_path))
            volume = sitk.GetArrayFromImage(data)

            label = record.polyp_class
            print('with label', label)

            for slice in record.slices:
                index = volume_index(record, slice)
                image = np.array(volume[index])
                image = resize(image)
                image = pre_process(image)

                feature = { 'train/label': _int64_feature(label),
                            'train/image': _float_feature(image.ravel())}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        writer.close()

    def generate_train_test(self, train_test_split):
        train, test = self._split(train_test_split)

        print('Creating train data')
        self._generate_tfrecords(train, self.train_writer)

        print('Creating test data')
        self._generate_tfrecords(test, self.test_writer)
