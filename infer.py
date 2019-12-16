import os
import tensorflow as tf

from train_util import *
from pipeline import *

class Infer():
    def __init__(self, args, model):
        self.args = args
        self.global_step = tf.Variable(0, trainable=False)
        self.test_data = os.path.join(args.base, args.test_datapath)
        self.test_size = len(list(tf.python_io.tf_record_iterator(self.test_data)))

        self.network = model(args.feature_shape, self.global_step, attention=args.attention)
        self.saver = tf.train.Saver()
        self.model_save_path = os.path.join(args.base, args.model_path)
        print(self.model_save_path)

    def infer(self):
        # Dataset pipeline
        pipeline = Pipeline(self.args.decode_record, self.args.record_shape)
        iterator_te = pipeline.create_test(self.test_data, self.test_size)
        iterator_te_next = iterator_te.get_next()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(iterator_te.initializer)
            batch_images, batch_labels = sess.run(iterator_te_next)
            binary_labels = binarise_labels(batch_labels)

            tf.global_variables_initializer().run()
            self.saver.restore(sess, self.model_save_path)
            test_accuracy(sess, self.network, iterator_te, iterator_te_next, self.args.feature_shape)
