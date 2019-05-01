import tensorflow as tf
from model.classifier import Classifier

class VGG(Classifier):
    def vgg_layer(self, net, out_channels, filter_dims=(3, 3), filter_strides=(1, 1),
                  pooling=False, padding='SAME', act_f = tf.nn.relu):
        net = tf.layers.conv2d(net, out_channels, filter_dims, strides=filter_strides, padding=padding, data_format="channels_first")
        net = tf.layers.batch_normalization(net, axis=1)
        if pooling:
            net = tf.layers.max_pooling2d(net, 2, 2, data_format='channels_first')
        net = act_f(net)
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

        net = self.batch_features

        net = self.vgg_layer(net, 256, pooling=True)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 256, pooling=True)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 256)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 256, pooling=True)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 512)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 512, pooling=True)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 512)
        print('--------------', net.shape)
        net = self.vgg_layer(net, 512, pooling=True)
        print('--------------', net.shape)

        net = tf.layers.flatten(net)
        net = self.dense_layer(net, 4096)
        net = self.dense_layer(net, 2048)
        net = self.dense_layer(net, 2, act_f=None, dropout=False)

        return self.build(net)
