import pydicom
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from main_util import *
from data.augmentation.augment_data import *
from pipeline import *
from model.vgg import VGG

# Paths
train_data = 'data/tfrecords/train_volume_balanced.tfrecords'
test_data = 'data/tfrecords/test_volume_balanced.tfrecords'
logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/logdir', str(datetime.now()))

# Parameters
input_shape = (5, 256, 256)
test_evaluation_period = 10
last_accuracy = test_evaluation_period

# Hyperparameters
batch_size = 64
test_size = min(batch_size, len(list(tf.python_io.tf_record_iterator(test_data))))
weight_decay = 1e-5
dropout_train_prob = 0.5

starter_learning_rate = 0.0001
N_steps_before_decay = 250
lr_decay_rate = 0.75
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           N_steps_before_decay, lr_decay_rate, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

# Dataset pipeline
pipeline = Pipeline(train_data, test_data)
iterator, iterator_te = pipeline.create(input_shape, batch_size, test_size)

# Initialise classification network
network = VGG(input_shape, learning_rate, weight_decay, global_step)

# Initialise augmentation
augmentor = Augmentor()

# Accuracy summary
accuracy_placeholder = tf.placeholder(tf.float32)
accuracy_summary = tf.summary.scalar('accuracy', accuracy_placeholder)

# Train
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        # Initialise variables
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        sess.run(iterator_te.initializer)

        # Summary writers
        summary_writer_tr = tf.summary.FileWriter(logdir + 'train_loss', sess.graph)
        summary_writer_te = tf.summary.FileWriter(logdir + 'test_loss', sess.graph)
        accuracy_writer_tr = tf.summary.FileWriter(logdir + 'train_accuracy', sess.graph)
        accuracy_writer_te = tf.summary.FileWriter(logdir + 'test_accuracy', sess.graph)

        batch = 0
        train_accuracies = []
        while (True):
            # Evaluate performance on test set at intervals
            if batch % test_evaluation_period == 0:
                summary_te, average_accuracy = test_accuracy(sess, network, batch, iterator_te, augmentor)
                summary_writer_te.add_summary(summary_te, int(batch))
                a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: average_accuracy})
                accuracy_writer_te.add_summary(a_s, int(batch))

            # Train the network
            batch_images, batch_labels = sess.run(iterator.get_next())
            aug_batch_images, aug_batch_labels = augment_batch(augmentor, batch_images, batch_labels)

            _, loss, summary, preds = sess.run([network.train_op, network.summary_loss, network.summary, network.predictions],
                                        feed_dict={network.batch_features: aug_batch_images,
                                                   network.batch_labels: aug_batch_labels,
                                                   network.dropout_prob: dropout_train_prob})

            # Summaries and statistics
            summary_writer_tr.add_summary(summary, int(batch))

            train_accuracies.append(accuracy(aug_batch_labels, preds))
            running_accuracy = np.average(train_accuracies[-last_accuracy:])

            a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: running_accuracy})
            accuracy_writer_tr.add_summary(a_s, int(batch))

            print('Train epoch %d.%d' % (batch // test_evaluation_period, batch % test_evaluation_period))
            print_statistics(loss, running_accuracy, prediction_class_balance(preds))

            batch += 1
