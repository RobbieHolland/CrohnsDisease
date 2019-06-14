import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score
from main_util import *
from augmentation.augment_data import *
from pipeline import *
from hard_soft_map import generate_batch_maps

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
        self.attention = args.attention
        self.mixedAttention = args.mixedAttention
        self.localisation = args.localisation
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size
        self.test_size = min(self.batch_size, len(list(tf.python_io.tf_record_iterator(self.test_data))))

        # Hyperparameters
        self.weight_decay = 0#5e-5
        self.dropout_train_prob = 0.5
        starter_learning_rate = 5e-6
        self.global_step = tf.Variable(0, trainable=False)
        # N_steps_before_decay = 1# 8000 // self.batch_size
        # end_learning_rate = 1e-6
        # lr_decay_rate = (end_learning_rate / starter_learning_rate) ** (1 / self.num_batches)
        # print(lr_decay_rate)
        # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
        #                                                 N_steps_before_decay, lr_decay_rate, staircase=True)

        # boundaries = [100]
        # values = [starter_learning_rate, 0.5 * starter_learning_rate]
        # self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
        self.learning_rate = starter_learning_rate

        # Logging
        self.best = {'batch': None, 'report': None, 'preds': None, 'loss': float("inf")}

    def write_log(self, line):
        with open(os.path.join(self.logdir, 'LOG'), 'a') as levels:
            levels.write(f'{line}\n')

    def update_stats(self, batch, loss, preds, labels):
        if loss < self.best['loss']:
            self.best['batch'] = batch
            self.best['loss'] = loss
            self.best['preds'] = preds
            self.best['labels'] = labels
            self.best['report'] = report(labels, preds)

    def create_summary(self, name, graph):
        path = os.path.join(self.logdir, f'fold{self.fold}', name)
        return tf.summary.FileWriter(path, graph)

    def log_metrics(self, sess, batch, writer, accuracy, f1):
        a_s = sess.run(self.accuracy_summary, feed_dict={self.accuracy_placeholder: accuracy})
        writer.add_summary(a_s, int(batch))

        f1_s = sess.run(self.f1_summary, feed_dict={self.f1_placeholder: f1})
        writer.add_summary(f1_s, int(batch))

    def train(self):
        # Dataset pipeline
        pipeline = Pipeline(self.decode_record, self.train_data, self.test_data)
        iterator, iterator_te = pipeline.create(self.record_shape, self.batch_size, self.test_size)
        iterator_next, iterator_te_next = iterator.get_next(), iterator_te.get_next()

        # Initialise classification network
        network = self.model(self.feature_shape, self.learning_rate, self.weight_decay,
                        self.global_step, self.attention, self.localisation, self.mixedAttention)

        # Initialise augmentation
        augmentor = Augmentor(self.feature_shape)

        # Summaries
        self.accuracy_placeholder = tf.placeholder(tf.float32)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy_placeholder)
        self.f1_placeholder = tf.placeholder(tf.float32)
        self.f1_summary = tf.summary.scalar('f1', self.f1_placeholder)

        # Train
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            # Initialise variables
            tf.global_variables_initializer().run()
            sess.run(iterator.initializer)
            sess.run(iterator_te.initializer)

            # Summary writers
            summary_writer_tr = self.create_summary('train', sess.graph)
            summary_writer_te = self.create_summary('test', sess.graph)

            train_accuracies = []
            for batch in range(self.num_batches):
                # Evaluate performance on test set at intervals
                if batch % self.test_evaluation_period == 0:
                    summary_te, overall_loss, preds, all_labels = test_accuracy(sess, network, batch, iterator_te, iterator_te_next, augmentor, self.feature_shape)
                    summary_writer_te.add_summary(summary_te, int(batch))
                    if not self.localisation:
                        overall_accuracy = accuracy(all_labels, preds)
                        self.update_stats(batch, overall_loss, preds, all_labels)
                        self.log_metrics(sess, batch, summary_writer_te, overall_accuracy, f1_score(all_labels, preds))
                        print_statistics(overall_loss, binarise_labels(all_labels), preds)
                    print('Loss: ', overall_loss)

                # Train the network
                batch_images, batch_labels, batch_coords = sess.run(iterator_next)
                aug_batch_images, batch_coords = augmentor.augment_batch(batch_images, batch_coords)
                hard_soft_maps = generate_batch_maps(batch_coords, np.array(self.feature_shape) // 2)

                binary_labels = binarise_labels(batch_labels)

                _, loss, summary, binary_preds = sess.run([network.train_op, network.summary_loss, network.summary, network.predictions],
                                            feed_dict={network.batch_features: aug_batch_images,
                                                       network.batch_labels: parse_labels(binary_labels),
                                                       network.batch_coords: batch_coords,
                                                       network.batch_hard_soft_maps: hard_soft_maps,
                                                       network.dropout_prob: self.dropout_train_prob})

                # Summaries and statistics
                print('-------- Train epoch %d.%d --------' % (batch // self.test_evaluation_period, batch % self.test_evaluation_period))
                summary_writer_tr.add_summary(summary, int(batch))

                if not self.localisation:
                    train_accuracies.append(accuracy(batch_labels, binary_preds))
                    running_accuracy = np.average(train_accuracies[-self.test_evaluation_period:])
                    self.log_metrics(sess, batch, summary_writer_tr, running_accuracy, f1_score(binary_labels, binary_preds))
                    print_statistics(loss, binary_labels, binary_preds)
                else:
                    print('Loss: ', loss)

            print('Training finished!')
            if not self.localisation:
                self.write_log(f'Best loss (epoch {self.best["batch"]}): {round(self.best["loss"], 3)}')
                self.write_log(f'with predictions: {self.best["preds"]}')
                self.write_log(f'of labels:        {self.best["labels"]}')
                self.write_log(self.best["report"])
                self.write_log('')
