import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from main_util import *
from augmentation.augment_data import *
from pipeline import *

class Runner:
    def __init__(self, args, model):
        # Paths
        self.logdir = os.path.join('/vol/gpudata/rh2515/CrohnsDisease/', args.logdir)
        self.fold = args.fold
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.train_data = args.train_datapath
        self.test_data = args.test_datapath
        self.write_log(f'Fold: {self.fold}')

        # Data processing
        self.decode_record = args.decode_record
        self.record_shape = args.record_shape

        # General parameters
        self.test_evaluation_period = 1
        self.num_batches = int(args.num_batches)

        # Network parameters
        self.model = model
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size
        self.test_size = min(self.batch_size, len(list(tf.python_io.tf_record_iterator(self.test_data))))

        # Hyperparameters
        self.weight_decay = 1e-4
        self.dropout_train_prob = 0.5
        starter_learning_rate = 0.00001
        N_steps_before_decay = 1000 // self.batch_size
        lr_decay_rate = 0.75
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        N_steps_before_decay, lr_decay_rate, staircase=True)

        # Logging
        self.best_accuracy = {'batch': None, 'accuracy': 0, 'preds': None}

    def write_log(self, line):
        with open(os.path.join(self.logdir, 'LOG'), 'a') as levels:
            levels.write(f'{line}\n')

    def update_stats(self, batch, test_accuracy, preds):
        if test_accuracy > self.best_accuracy['accuracy']:
            self.best_accuracy['batch'] = batch
            self.best_accuracy['accuracy'] = test_accuracy
            self.best_accuracy['preds'] = preds

    def create_summary(self, name, graph):
        path = os.path.join(self.logdir, f'fold{self.fold}', name)
        return tf.summary.FileWriter(path, graph)

    def train(self):
        # Dataset pipeline
        pipeline = Pipeline(self.decode_record, self.train_data, self.test_data)
        iterator, iterator_te = pipeline.create(self.record_shape, self.batch_size, self.test_size)
        iterator_next, iterator_te_next = iterator.get_next(), iterator_te.get_next()

        # Initialise classification network
        network = self.model(self.feature_shape, self.learning_rate, self.weight_decay, self.global_step)

        # Initialise augmentation
        augmentor = Augmentor(self.feature_shape)

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
                summary_writer_tr = self.create_summary('train_loss', sess.graph)
                summary_writer_te = self.create_summary('test_loss', sess.graph)
                accuracy_writer_tr = self.create_summary('train_accuracy', sess.graph)
                accuracy_writer_te = self.create_summary('test_accuracy', sess.graph)

                train_accuracies = []
                for batch in range(self.num_batches):
                    # Evaluate performance on test set at intervals
                    if batch % self.test_evaluation_period == 0:
                        summary_te, average_accuracy, preds = test_accuracy(sess, network, batch, iterator_te, iterator_te_next, self.feature_shape)
                        summary_writer_te.add_summary(summary_te, int(batch))
                        a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: average_accuracy})
                        accuracy_writer_te.add_summary(a_s, int(batch))
                        self.update_stats(batch, average_accuracy, preds)

                    # Train the network
                    batch_images, batch_labels = sess.run(iterator_next)
                    aug_batch_images = augmentor.augment_batch(batch_images)

                    parsed_batch_labels = parse_labels(batch_labels)

                    _, loss, summary, preds = sess.run([network.train_op, network.summary_loss, network.summary, network.predictions],
                                                feed_dict={network.batch_features: aug_batch_images,
                                                           network.batch_labels: parsed_batch_labels,
                                                           network.dropout_prob: self.dropout_train_prob})

                    # Summaries and statistics
                    print('-------- Train epoch %d.%d --------' % (batch // self.test_evaluation_period, batch % self.test_evaluation_period))
                    summary_writer_tr.add_summary(summary, int(batch))

                    train_accuracies.append(accuracy(parsed_batch_labels, preds))
                    running_accuracy = np.average(train_accuracies[-self.test_evaluation_period:])

                    a_s = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: running_accuracy})
                    accuracy_writer_tr.add_summary(a_s, int(batch))

                    print_statistics(loss, running_accuracy, prediction_class_balance(preds))

                print('Training finished!')
                self.write_log(f'Best accuracy (epoch {self.best_accuracy["batch"]}): {round(self.best_accuracy["accuracy"], 3)}%')
                self.write_log(f'with predictions: {self.best_accuracy["preds"]}')
                self.write_log('')
