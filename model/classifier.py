import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self, input_shape, lr, weight_decay, global_step, localisation, mixedAttention):
        self.batch_features = tf.placeholder(tf.float32, shape=(None,) + input_shape)
        self.batch_labels = tf.placeholder(tf.float64)
        self.batch_coords = tf.placeholder(tf.float32, shape=(None,3))
        self.batch_hard_soft_maps = tf.placeholder(tf.float32, shape=(None,) + tuple(np.array(input_shape) // 2))

        # Initialise Model
        self.lr = lr
        self.weight_decay = weight_decay
        self.global_step = global_step
        self.localisation = localisation
        self.mixedAttention = mixedAttention
        self.mixed_loss_coefficient = 1

    def localisation_loss(self, net_output):
        self.predictions = net_output
        loss = tf.reduce_mean(tf.squared_difference(net_output, self.batch_coords))
        return loss

    def classification_loss(self, net_output):
        self.predictions = tf.argmax(tf.nn.softmax(net_output), axis=1)
        ground_truth = tf.expand_dims(self.ground_truth, 1)

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ground_truth, logits=net_output)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = cross_entropy_loss + self.weight_decay * l2_loss

        self.cross_entropy = tf.reduce_mean(cross_entropy_loss)
        tf.summary.scalar("cross_entropy", self.cross_entropy)
        return loss

    def build(self, net_output, mixed_loss):
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.ground_truth = tf.cast(self.batch_labels, tf.float32)

        if self.localisation:
            loss = self.localisation_loss(net_output)
        else:
            loss = self.classification_loss(net_output)
        if self.mixedAttention:
            loss += self.mixed_loss_coefficient * mixed_loss

        self.train_op = self.optimiser.minimize(loss, global_step=self.global_step, colocate_gradients_with_ops=True)

        tf.summary.scalar("learning_rate", self.lr)
        self.summary_loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.summary_loss)

        self.summary = tf.summary.merge_all()
