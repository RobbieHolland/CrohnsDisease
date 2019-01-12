import tensorflow as tf

def vgg_layer(net, out_channels, filter_dims, strides, padding='SAME', act_f = tf.nn.relu):
    net = tf.layers.conv2d(net, out_channels, filter_dims, strides=strides, padding=padding)
    net = act_f(net)
    return net

class Model:
    def __init__(self, image_width, image_height, lr, batch_size=1):
        self.input_slices = tf.placeholder(tf.float32, (batch_size, image_height, image_width), name='input_slices')
        net = tf.expand_dims(self.input_slices, axis=3)

        net = vgg_layer(net, 64, (3, 3), (2, 2))
        net = vgg_layer(net, 128, (3, 3), (2, 2))
        net = vgg_layer(net, 256, (3, 3), (2, 2))
        net = vgg_layer(net, 512, (3, 3), (1, 1))
        net = vgg_layer(net, 512, (3, 3), (2, 2))
        net = vgg_layer(net, 512, (3, 3), (2, 2))
        net = vgg_layer(net, 512, (3, 3), (2, 2))
        net = tf.layers.flatten(net)
        self.prediction = tf.layers.dense(net, 1)

        self.ground_truth = tf.placeholder(tf.float32, (batch_size), name='labels')
        ground_truth = tf.expand_dims(self.ground_truth, 1)

        self.loss = tf.reduce_mean(tf.square(self.ground_truth - self.prediction))
        tf.summary.scalar("loss", self.loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        self.summary = tf.summary.merge_all()
