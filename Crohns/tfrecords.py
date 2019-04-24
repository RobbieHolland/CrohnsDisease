import os
import tensorflow as tf
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    # Since this will be used to convert an np.array we don't use []
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tfrecord_name(set, suffix=''):
    return f'{suffix}_{set}.tfrecords'

class TFRecordGenerator:
    def __init__(self, out_path, suffix):
        self.out_path = out_path
        self.suffix = suffix

    # Features: List of Sitk images
    # Labels: Corresponding list of labels
    def _generate_tfrecords(self, features, labels, writer):
        for i, (feature, label) in enumerate(zip(features, labels)):
            try:
                image_array = sitk.GetArrayFromImage(feature)
                feature = { 'train/label': _int64_feature(label),
                            'train/axial_t2': _float_feature(image_array.ravel())}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except Exception as e:
                print('Error generating record')
                print(e)
            print(f'{round(100 * i / len(labels), 3)}% \r', end='')
        writer.close()

    def generate_train_test(self, test_proportion, X, y):
        train_path = os.path.join(self.out_path, tfrecord_name('train', self.suffix))
        test_path = os.path.join(self.out_path, tfrecord_name('test', self.suffix))
        if os.path.isfile(train_path) or os.path.isfile(test_path):
            print(f'Train or test with suffix {self.suffix} already exists.')
            print('Press Enter to continue and overwrite.')
            input()

        self.train_writer = tf.python_io.TFRecordWriter(os.path.join(self.out_path, tfrecord_name('train', self.suffix)))
        self.test_writer  = tf.python_io.TFRecordWriter(os.path.join(self.out_path, tfrecord_name('test', self.suffix)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, stratify=y, random_state=0)

        print('Creating train data...')
        self._generate_tfrecords(X_train, y_train, self.train_writer)

        print('Creating test data...')
        self._generate_tfrecords(X_test, y_test, self.test_writer)
