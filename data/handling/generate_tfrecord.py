'''
Generates TFRecords of the dataset
'''

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
        self.metadata = self.reader.shuffle_read()

        self.metadata.print_statistics()

        self.train_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, 'train_all.tfrecords'))
        self.test_writer  = tf.python_io.TFRecordWriter(os.path.join(out_path, 'test_all.tfrecords'))

    def _generate_tfrecords(self, records, writer):
        for record in records:
            print('Loading', record.patient_no)
            data = sitk.ReadImage(record.form_path(self.data_path))
            volume = sitk.GetArrayFromImage(data)
            print(record.patient_position)
            print(volume.shape)
            label = record.polyp_class
            print('with label', label)

            for slice in record.slices:
                index = volume_index(record, slice)
                try:
                    image = np.array(volume[index])
                    image = resize(image)
                    image = pre_process(image)

                    feature = { 'train/label': _int64_feature(label),
                                'train/image': _float_feature(image.ravel())}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                except Exception as e:
                    print('Error generating record for', record.patient_no, record.patient_position)
                    print(e)

        writer.close()

    def generate_train_test(self, train_test_split):
        train, test = self.metadata.split(train_test_split)

        print('Creating train data...')
        self._generate_tfrecords(train, self.train_writer)

        print('Creating test data...')
        self._generate_tfrecords(test, self.test_writer)
