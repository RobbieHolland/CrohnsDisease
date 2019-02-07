import pydicom
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from data.augmentation.augment_data import *
from model.vgg import Model

# Paths
train_data = 'data/tfrecords/train.tfrecords'
test_data = 'data/tfrecords/test.tfrecords'
logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/logdir', str(datetime.now()))

# Parameters
image_width = 256
image_height = 256
test_size = len(list(tf.python_io.tf_record_iterator(test_data)))
test_evaluation_period = 8

# Hyperparameters
batch_size = 32
learning_rate = 0.0001

# Dataset pipeline
def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'train/image': tf.FixedLenFeature([image_width, image_height], tf.float32),
                  'train/label': tf.FixedLenFeature([], tf.int64)})

    return features['train/image'], features['train/label']

# Train pipeline
dataset = tf.data.TFRecordDataset(train_data).map(decode)
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

# Dataset iterator
iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()

# Initialise Model
augmented_features = tf.placeholder(tf.float32, shape=(None, 256, 256))
augmented_labels = tf.placeholder(tf.float64)

network = Model(augmented_features, augmented_labels, image_width, image_height, learning_rate, batch_size)
init = tf.global_variables_initializer()

# Initialise augmentation
augmentor = Augmentor()

# Test pipeline
dataset_te = tf.data.TFRecordDataset(test_data).map(decode)
dataset_te = dataset_te.batch(test_size)
dataset_te = dataset_te.repeat(None)
iterator_te = dataset_te.make_initializable_iterator()
features_te, labels_te = iterator_te.get_next()

def augment_batch(features, labels):
    aug_batch_images = augmentor.augment(batch_images)
    aug_batch_labels = [[0, 1] if polyp > 0 else [1, 0] for polyp in batch_labels]
    return aug_batch_images, aug_batch_labels

# Train
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        # Initialise variables
        sess.run(init)
        sess.run(iterator.initializer)
        sess.run(iterator_te.initializer)

        summary_writer_tr = tf.summary.FileWriter(logdir + 'train', sess.graph)
        summary_writer_te = tf.summary.FileWriter(logdir + 'test', sess.graph)

        batch = 0
        while (True):
            batch_images, batch_labels = sess.run([features, labels])
            aug_batch_images, aug_batch_labels = augment_batch(batch_images, batch_labels)

            _, loss, summary = sess.run([network.train_op, network.summary_loss, network.summary],
                                        feed_dict={augmented_features: aug_batch_images,
                                                   augmented_labels: aug_batch_labels})

            print(batch // test_evaluation_period, '-', batch % test_evaluation_period, ':', 'Loss', loss)
            summary_writer_tr.add_summary(summary, int(batch))

            batch += 1
            if batch % test_evaluation_period == 0:
                batch_images, batch_labels = sess.run([features_te, labels_te])
                aug_batch_images, aug_batch_labels = augment_batch(batch_images, batch_labels)

                loss_te, summary_te = sess.run([network.summary_loss, network.summary],
                                                feed_dict={augmented_features: aug_batch_images,
                                                           augmented_labels: aug_batch_labels})
                print('--- Test Loss  ', loss_te)
                summary_writer_te.add_summary(summary_te, int(batch))
                pass
