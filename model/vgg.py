import tensorflow as tf

def vgg_layer(net, out_channels, filter_dims=(3, 3), filter_strides=(1, 1),
              pooling=False, padding='SAME', act_f = tf.nn.relu):
    net = tf.layers.conv2d(net, out_channels, filter_dims, strides=filter_strides, padding=padding)
    if pooling:
        net = tf.layers.max_pooling2d(net, 2, 2)
    net = act_f(net)
    return net

class Model:
    def __init__(self, batch_features, batch_labels, image_width, image_height, lr, batch_size=1):
        net = tf.expand_dims(batch_features, axis=3)

        net = vgg_layer(net, 64, pooling=True)
        net = vgg_layer(net, 128, pooling=True)
        net = vgg_layer(net, 256)
        net = vgg_layer(net, 256, pooling=True)
        net = vgg_layer(net, 512)
        net = vgg_layer(net, 512, pooling=True)
        net = vgg_layer(net, 512)
        net = vgg_layer(net, 512, pooling=True)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 2)

        self.ground_truth = tf.cast(batch_labels, tf.float32)
        ground_truth = tf.expand_dims(self.ground_truth, 1)

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ground_truth, logits=net)

        self.summary_loss = tf.reduce_mean(cross_entropy_loss)
        tf.summary.scalar("loss", self.summary_loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_loss)

        self.summary = tf.summary.merge_all()
