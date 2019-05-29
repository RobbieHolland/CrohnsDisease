import tensorflow as tf
from model.classifier import Classifier
Conv3D = tf.layers.conv3d

# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#522

def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)

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

    def build_resnet(self, net, filters):
        def projection_shortcut(net, out_channels, filter_strides, padding='SAME'):
            return Conv3D(net, out_channels, 1, strides=filter_strides, padding=padding, data_format="channels_first")

        for channels, stride in filters:
            net = self.block(net, channels, projection_shortcut, filter_strides=stride)
            print(net.shape)
        return net

    def dense_layer(self, net, n_outputs, act_f = tf.nn.relu, dropout=True):
        net = tf.layers.dense(net, n_outputs)
        if act_f != None:
            net = act_f(net)
        if dropout:
            net = tf.nn.dropout(net, self.dropout_prob)
        return net

    def build_ti_net(self, input):
        filters = [(64, 2), (64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1), (256, 1)]
        net = self.build_resnet(input, filters)
        print(net.shape)
        net = tf.layers.average_pooling3d(net, net.shape[2:], 1, padding='valid', data_format='channels_first')
        print(net.shape)
        net = tf.layers.flatten(net)
        net = self.dense_layer(net, 2, act_f=None, dropout=True)
        print(net.shape)
        return net

    def build_long_net(self, input):
        filters = [(32, 2), (32, 1), (32, 1), (64, 2), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1)]
        net = self.build_ti_net(input, filters)
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
        net = make_parallel(self.build_ti_net, 1, input=net)

        self.build(net)
