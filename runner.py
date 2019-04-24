import pydicom
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from main_util import *
from augmentation.augment_data import *
from pipeline import *
from model.vgg import VGG

class Runner:
    def __init__(self, args):
        # Paths
        self.logdir = os.path.join('/vol/bitbucket/rh2515/CrohnsDisease/',
                                    args.logdir, str(datetime.now()))
        self.train_data = args.train_datapath
        self.test_data = args.test_datapath

        # Data processing
        self.decode_record = args.decode_record

        # General parameters
        self.test_evaluation_period = 10

        # Network parameters
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size
        self.test_size = min(self.batch_size, len(list(tf.python_io.tf_record_iterator(self.test_data))))

        # Hyperparameters
        self.weight_decay = 1e-4
        self.dropout_train_prob = 0.5
        starter_learning_rate = 0.0001
        N_steps_before_decay = 250
        lr_decay_rate = 0.75
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        N_steps_before_decay, lr_decay_rate, staircase=True)

    def train(self):
        # Dataset pipeline
        pipeline = Pipeline(self.decode_record, self.train_data, self.test_data)
        iterator, iterator_te = pipeline.create(self.feature_shape, self.batch_size, self.test_size)
        iterator_next, iterator_te_next = iterator.get_next(), iterator_te.get_next()

        # Initialise classification network
        network = VGG(self.feature_shape, self.learning_rate, self.weight_decay, self.global_step)

        # Initialise augmentation
        augmentor = Augmentor()

        # Summaries
        tf.summary.scalar('learning_rate', self.learning_rate)
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
                summary_writer_tr = tf.summary.FileWriter(self.logdir + 'train_loss', sess.graph)
                summary_writer_te = tf.summary.FileWriter(self.logdir + 'test_loss', sess.graph)
                accuracy_writer_tr = tf.summary.FileWriter(self.logdir + 'train_accuracy', sess.graph)
                accuracy_writer_te = tf.summary.FileWriter(self.logdir + 'test_accuracy', sess.graph)

                batch = 0
                train_accuracies = []
                while (True):
                    # Evaluate performance on test set at intervals
                    if batch % self.test_evaluation_period == 0:
                        summary_te, average_accuracy = test_accuracy(sess, network, batch, iterator_te, iterator_te_next, augmentor)
                        summary_writer_te.add_summary(summary_te, int(batch))
                        a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: average_accuracy})
                        accuracy_writer_te.add_summary(a_s, int(batch))

                    # Train the network
                    batch_images, batch_labels = sess.run(iterator_next)
                    aug_batch_images, aug_batch_labels = augment_batch(augmentor, batch_images, batch_labels)

                    _, loss, summary, preds = sess.run([network.train_op, network.summary_loss, network.summary, network.predictions],
                                                feed_dict={network.batch_features: aug_batch_images,
                                                           network.batch_labels: aug_batch_labels,
                                                           network.dropout_prob: self.dropout_train_prob})

                    # Summaries and statistics
                    summary_writer_tr.add_summary(summary, int(batch))

                    train_accuracies.append(accuracy(aug_batch_labels, preds))
                    running_accuracy = np.average(train_accuracies[-self.test_evaluation_period:])

                    a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: running_accuracy})
                    accuracy_writer_tr.add_summary(a_s, int(batch))

                    print('Train epoch %d.%d' % (batch // self.test_evaluation_period, batch % self.test_evaluation_period))
                    print_statistics(loss, running_accuracy, prediction_class_balance(preds))

                    batch += 1
