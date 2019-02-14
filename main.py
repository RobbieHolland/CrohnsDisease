import pydicom
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from data.augmentation.augment_data import *
from pipeline import *
from model.vgg import Model

# Paths
train_data = 'data/tfrecords/train_all.tfrecords'
test_data = 'data/tfrecords/test_all.tfrecords'
logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/logdir', str(datetime.now()))

# Parameters
image_width = 256
image_height = 256
test_evaluation_period = 16
last_accuracy = test_evaluation_period

# Hyperparameters
batch_size = 32
test_size = min(batch_size, len(list(tf.python_io.tf_record_iterator(test_data))))
learning_rate = 0.0001

# Dataset pipeline
pipeline = Pipeline(train_data, test_data)
iterator, iterator_te = pipeline.create(image_width, image_height, batch_size, test_size)
features, labels = iterator.get_next()
features_te, labels_te = iterator_te.get_next()

# Initialise Model
augmented_features = tf.placeholder(tf.float32, shape=(None, 256, 256))
augmented_labels = tf.placeholder(tf.float64)

network = Model(augmented_features, augmented_labels, image_width, image_height, learning_rate)
init = tf.global_variables_initializer()

# Initialise augmentation
augmentor = Augmentor()

# Show test data
test_images = tf.summary.image('Test images', tf.expand_dims(features_te, -1), max_outputs=test_size)

def augment_batch(features, labels):
    aug_batch_images = augmentor.augment(features)
    aug_batch_labels = [[0, 1] if polyp > 0 else [1, 0] for polyp in labels]
    return aug_batch_images, aug_batch_labels

def accuracy(labels, preds):
    arg_labels = np.argmax(labels, axis=1)
    return np.sum(np.array(arg_labels) == np.array(preds)) / len(labels)

def test_accuracy(sess, batch):
    accuracies = []
    losses = []
    summary_te = None

    try:
        while (True):
            batch_images, batch_labels = sess.run([features_te, labels_te])
            aug_batch_images, aug_batch_labels = augment_batch(batch_images, batch_labels)

            loss_te, summary_te, preds = sess.run([network.summary_loss, network.summary, network.predictions],
                                            feed_dict={augmented_features: aug_batch_images,
                                                       augmented_labels: aug_batch_labels})
            accuracies.append(accuracy(aug_batch_labels, preds))
            losses.append(loss_te)

    except tf.errors.OutOfRangeError:
        summary_writer_te.add_summary(summary_te, int(batch))
        sess.run(iterator_te.initializer)
        print()
        print('Test Loss:     ', np.average(losses))
        print('Test accuracy: ', np.average(accuracies))
        print()


# Train
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        # Initialise variables
        sess.run(init)
        sess.run(iterator.initializer)
        sess.run(iterator_te.initializer)

        # Summary writers
        summary_writer_tr = tf.summary.FileWriter(logdir + 'train', sess.graph)
        summary_writer_te = tf.summary.FileWriter(logdir + 'test', sess.graph)

        # Show test data
        t_s = sess.run(test_images)
        summary_writer_te.add_summary(t_s)

        batch = 0
        train_accuracies = []
        while (True):
            if batch % test_evaluation_period == 0:
                test_accuracy(sess, batch)

            batch_images, batch_labels = sess.run([features, labels])
            aug_batch_images, aug_batch_labels = augment_batch(batch_images, batch_labels)

            _, loss, summary, preds = sess.run([network.train_op, network.summary_loss, network.summary, network.predictions],
                                        feed_dict={augmented_features: aug_batch_images,
                                                   augmented_labels: aug_batch_labels})

            print(batch // test_evaluation_period, '-', batch % test_evaluation_period, ':', 'Loss', loss)
            train_accuracies.append(accuracy(aug_batch_labels, preds))
            print('Train accuracy:', np.average(train_accuracies[-last_accuracy:]))
            summary_writer_tr.add_summary(summary, int(batch))

            batch += 1
