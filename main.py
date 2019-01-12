import pydicom
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import os
from data.data_handler import DataHandler
from model.vgg import Model

# Parameters
logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/logdir', str(datetime.now()))
image_width = 256
image_height = 256

# Hyperparameters
portion_train = 0.8
batch_size = 16
learning_rate = 0.001

# Load data into memory (dataset small)
data_handler = DataHandler('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
features, labels = data_handler.load_dataset()

ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.shuffle(len(features))
ds = ds.batch(batch_size)
ds_iter = ds.make_initializable_iterator()
next_batch = ds_iter.get_next()

print('Loaded images into memory')

# Initialise Model
network = Model(image_width, image_height, learning_rate, batch_size)
init = tf.global_variables_initializer()

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        epoch = 0
        while (True):
            sess.run(ds_iter.initializer)
            batch = sess.run(next_batch)

            _, loss, summary = sess.run([network.train_op, network.loss, network.summary],
                                        feed_dict=  {network.input_slices: batch[0],
                                                    network.ground_truth: batch[1]})

            epoch += 1
            summary_writer.add_summary(summary, int(epoch))
