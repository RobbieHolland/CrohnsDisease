import tensorflow as tf
from model.classifier import Classifier
Conv3D = tf.layers.conv3d

# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#522
class ResNet3D(Classifier):
    def block(self, net, out_channels, shortcut_f, filter_dims=3, filter_strides=2,
                  padding='SAME', act_f=tf.nn.relu):
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        shortcut = shortcut_f(net, out_channels, filter_strides)

        net = Conv3D(net, out_channels, filter_dims, strides=filter_strides, padding=padding, data_format="channels_first")
        net = tf.layers.batch_normalization(net, axis=1)
        net = tf.nn.relu(net)
        net = Conv3D(net, out_channels, filter_dims, strides=1, padding=padding, data_format="channels_first")

        net = net + shortcut
        net = tf.layers.batch_normalization(net, axis=1)
        net = tf.nn.relu(net)

        return net

    def build_resnet(self, net):
        def projection_shortcut(net, out_channels, filter_strides, padding='SAME'):
            return Conv3D(net, out_channels, 1, strides=filter_strides, padding=padding, data_format="channels_first")

        filter_sizes = [8, 16, 32, 16, 8]
        for filters in filter_sizes:
            net = self.block(net, filters, projection_shortcut)
            print(net.shape)
        return net

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
        print('shape', self.batch_features.shape)

        net = tf.expand_dims(self.batch_features, axis=1)
        print(net.shape)
        net = self.build_resnet(net)
        print(net.shape)
        net = tf.layers.flatten(net)
        net = self.dense_layer(net, 2, act_f=None, dropout=True)
        print(net.shape)

        self.build(net)
