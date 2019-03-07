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
    height = image.shape[0]
    itk_image = sitk.GetImageFromArray(np.array(image))
    resample = sitk.ResampleImageFilter()
    scaled_image = resample.Execute(
        itk_image,
        (256, 256, height),
        sitk.ScaleTransform(3, (2, 2, 1)),
        sitk.sitkLinear,
        itk_image.GetOrigin(),
        itk_image.GetSpacing(),
        itk_image.GetDirection(),
        0,
        itk_image.GetPixelID())
    scaled_image = sitk.GetArrayFromImage(scaled_image)

    return scaled_image

def pre_process(image):
    # Normalise image
    image = whitening(image)
    return image

def volume_index(volume_height, slice):
    return volume_height - 1 - slice

def tfrecord_name(set, suffix=''):
    return set + '_' + suffix + '.tfrecords'

def centroid_volume(record, centroid, whole_volume):
    volume_height = whole_volume.shape[0]
    bottom, top = centroid - record.neighbour_distance, centroid + record.neighbour_distance
    vol = ()
    for slice in range(bottom, top + 1):
        index = min(max(0, volume_index(volume_height, slice)), volume_height)
        vol += (whole_volume[index],)

    vol = np.stack(vol)
    print(vol.shape)
    print(vol.size * vol.itemsize)
    return vol

class TFRecordGenerator:
    def __init__(self, label_path, data_path, out_path):
        self.out_path = out_path
        self.reader = DataParser(label_path, data_path)
        self.data_path = data_path
        self.metadata = self.reader.shuffle_read()

        self.metadata.print_statistics()

    def _generate_tfrecords(self, metadata, writer):
        metadata.print_statistics()
        for record in metadata.records:
            # For 2D task, only care about slices - for 3D task this is not required
            if record.n_centroids() == 0:
                continue

            print('Loading', record.patient_no)
            data = sitk.ReadImage(record.form_path(self.data_path))
            volume = sitk.GetArrayFromImage(data)
            if volume.shape[0] < 50:
                print('Skipping', record.patient_no)
                continue
            print(record.patient_position, volume.shape)
            label = record.polyp_class
            print('with label', label)

            for centroid in record.slice_centroids:
                try:
                    # index = volume_index(record, centroid)
                    image = centroid_volume(record, centroid, volume)
                    # image = np.array(volume[index])
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

    def generate_train_test(self, train_test_split, suffix):
        train_path = os.path.join(self.out_path, tfrecord_name('train', suffix))
        test_path = os.path.join(self.out_path, tfrecord_name('test', suffix))
        if os.path.isfile(train_path) or os.path.isfile(test_path):
            print('Train or test with suffix', suffix, 'already exists.')
            print('Press Enter to proceed anyway.')
            input()

        self.train_writer = tf.python_io.TFRecordWriter(os.path.join(self.out_path, tfrecord_name('train', suffix)))
        self.test_writer  = tf.python_io.TFRecordWriter(os.path.join(self.out_path, tfrecord_name('test', suffix)))
        train, test = self.metadata.split(train_test_split)

        print('Creating train data...')
        self._generate_tfrecords(train, self.train_writer)

        print('Creating test data...')
        self._generate_tfrecords(test, self.test_writer)
