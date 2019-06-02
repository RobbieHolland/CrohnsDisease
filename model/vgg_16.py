import tensorflow as tf
from model.classifier import Classifier
Conv3D = tf.layers.conv3d

class VGG16(Classifier):
    def vgg_layer(self, net, channels, filter_size=3, filter_strides=1, padding='SAME', df='channels_first'):
        net = Conv3D(net, channels, filter_size, strides=filter_strides, padding=padding, data_format=df)
        net = tf.layers.batch_normalization(net, axis=1)
        net = tf.nn.relu(net)
        return net

    def dense_layer(self, net, n_outputs, act_f = tf.nn.relu, dropout=True):
        net = tf.layers.dense(net, n_outputs)
        if act_f != None:
            net = act_f(net)
        if dropout:
            net = tf.nn.dropout(net, self.dropout_prob)
        return net

    def build_vgg(self, net):
        layers  = [(64, 3), (64, 3), 'maxpool', (128, 3), (128, 3), 'maxpool', (256, 3), (256, 3)] \
                + [(256, 1), 'maxpool', (512, 3), (512, 3), (512, 1), 'maxpool', (512, 3), (512, 3), (512, 1), 'maxpool']

        for layer in layers:
            if layer == 'maxpool':
                net = tf.layers.max_pooling3d(net, 2, 2, data_format='channels_first')
            else:
                channels, filter_size = layer
                net = self.vgg_layer(net, channels, filter_size=filter_size)
            print(net.shape)


        net = tf.layers.average_pooling3d(net, net.shape[2:], 1, padding='valid', data_format='channels_first')
        print(net.shape)
        net = tf.layers.flatten(net)
        net = self.dense_layer(net, 2, act_f=None, dropout=True)
        print(net.shape)
        return net

    def __init__(self, input_shape, lr, weight_decay, global_step):
        super().__init__(input_shape, lr, weight_decay, global_step)

        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())
        print('shape', self.batch_features.shape)

        net = tf.expand_dims(self.batch_features, axis=1)
        print(net.shape)

        # net = make_parallel(self.build_long_net, 1, input=net)
        net = self.build_vgg(net)

        self.build(net)
