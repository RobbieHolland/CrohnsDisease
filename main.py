import pydicom
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from data.handling.data_handler import DataHandler
from data.augmentation.augment_data import Augmentor
from model.vgg import Model

# Parameters
logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/logdir', str(datetime.now()))
image_width = 256
image_height = 256
test_evaluation_period = 3

# Hyperparameters
batch_size = 54
learning_rate = 0.00001
portion_test = 0.15

# Data iterator
features = tf.placeholder(tf.float32, (None, image_width, image_height))
labels = tf.placeholder(tf.int32, (None))

ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.shuffle(batch_size)
ds = ds.batch(batch_size)
ds_iter = ds.make_initializable_iterator()
next_batch = ds_iter.get_next()

print('Loaded images into memory')

# Initialise Model
network = Model(next_batch, image_width, image_height, learning_rate, batch_size)
init = tf.global_variables_initializer()

# Initialise augmentation
augmentor = Augmentor()

# Load data into memory (dataset small)
data_handler = DataHandler('/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY', './data/cases/index')
features_tr, labels_tr, features_te, labels_te = data_handler.load_dataset(portion_test)

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        sess.run(ds_iter.initializer, feed_dict = {features: features_tr, labels: labels_tr})
        summary_writer_tr = tf.summary.FileWriter(logdir + 'train', sess.graph)
        summary_writer_te = tf.summary.FileWriter(logdir + 'test', sess.graph)

        epoch, batch = 0, 0
        while (True):
            try:
                batch += 1
                _, loss, summary = sess.run([network.train_op, network.loss, network.summary])

                print('Epoch ', epoch, '- Batch ', batch)
                print('Loss  ', loss)
                summary_writer_tr.add_summary(summary, int(batch))

            # Dataset iteration complete
            except tf.errors.OutOfRangeError:
                epoch += 1
                if epoch % test_evaluation_period == 0:
                    sess.run(ds_iter.initializer, feed_dict = {features: features_te, labels: labels_te})
                    loss_te, summary_te = sess.run([network.loss, network.summary])
                    print('--- Test Loss  ', loss_te)
                    summary_writer_te.add_summary(summary_te, int(batch))

                # Initialise dataset with newly augmented data
                augmented_slices = augmentor.augment(features_tr)
                sess.run(ds_iter.initializer, feed_dict = {features: features_tr, labels: labels_tr})
