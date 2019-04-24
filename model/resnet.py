import tensorflow as tf
from model.classifier import Classifier

class ResNet(Classifier):
    def projection_shortcut(net, out_channels, padding='SAME'):
        return tf.layers.conv2d(net, out_channels, (1, 1), strides=(1, 1), padding=padding, data_format="channels_first")

    def block(self, net, out_channels, filter_dims=(3, 3), filter_strides=(1, 1),
                  pooling=False, padding='SAME', act_f=tf.nn.relu, shortcut_f=projection_shortcut):
        shortcut = net
        net = tf.layers.batch_normalization(net, axis=1)
        net = act_f(net)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = shortcut_f(net)

        net = tf.layers.conv2d(net, out_channels, filter_dims, strides=filter_strides, padding=padding, data_format="channels_first")
        net = tf.layers.batch_normalization(net, axis=1)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, out_channels, filter_dims, strides=(1, 1), padding=padding, data_format="channels_first")

        return net + shortcut

    def build_resnet(self):
        # https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#522


    def dense_layer(self, net, n_outputs, act_f = tf.nn.relu, dropout=True):
        net = tf.layers.dense(net, n_outputs)
        if act_f != None:
            net = act_f(net)
        if dropout:
            net = tf.nn.dropout(net, self.dropout_prob)
        return net

    def __init__(self, input_shape, lr, weight_decay, global_step):
        super().__init__(input_shape, lr, weight_decay, global_step)

        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())

        net = tf.transpose(self.batch_features, perm=[0, 2, 3, 1])

        net = self.vgg_layer(net, 256, pooling=True)
        net = self.vgg_layer(net, 256, pooling=True)
        net = self.vgg_layer(net, 256)
        net = self.vgg_layer(net, 256, pooling=True)
        net = self.vgg_layer(net, 512)
        net = self.vgg_layer(net, 512, pooling=True)
        net = self.vgg_layer(net, 512)
        net = self.vgg_layer(net, 512, pooling=True)

        net = tf.layers.flatten(net)
        net = self.dense_layer(net, 4096)
        net = self.dense_layer(net, 2048)
        net = self.dense_layer(net, 2, act_f=None, dropout=False)

        return self.build(net)
