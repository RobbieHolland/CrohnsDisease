import tensorflow as tf
from model.classifier import Classifier
from model.attention_layer import GridAttentionBlock
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

    def dense_layer(self, net, n_outputs, act_f = tf.nn.relu, dropout=True):
        net = tf.layers.dense(net, n_outputs)
        if act_f != None:
            net = act_f(net)
        if dropout:
            net = tf.nn.dropout(net, self.dropout_prob)
        return net

    def n_convs(self, net, specs):
        for channels, stride in specs:
            net = self.block(net, channels, self.p_s, filter_strides=stride)
            print(net.shape)
        return net

    def classify(self, net):
        pooled = tf.layers.average_pooling3d(net, net.shape[2:], 1, padding='valid', data_format='channels_first')
        net = tf.layers.flatten(pooled)
        print('flattened', net.shape)
        net = self.dense_layer(net, 2, act_f=None, dropout=True)
        return net

    def weighted_prediction(self, global_logits, attention_logits_s1, attention_logits_s2):
        global_weight = tf.Variable(0.5, trainable=True)
        attention_weight_s1 = tf.Variable(0.25, trainable=True)
        attention_weight_s2 = tf.Variable(0.25, trainable=True)
        tf.summary.scalar('global_weight', global_weight)
        tf.summary.scalar('attention_weight_s1', attention_weight_s1)
        tf.summary.scalar('attention_weight_s2', attention_weight_s2)
        attention_prediction = attention_weight_s1 * attention_logits_s1 + attention_weight_s2 * attention_logits_s2
        return global_weight * global_logits + attention_prediction

    def max_prediction(self, global_logits, attention_logits_s1, attention_logits_s2):
        return tf.reduce_max(tf.stack((global_logits, attention_logits_s1, attention_logits_s2), axis=0), axis=0)

    def build_ti_net(self, batch, attention=False, localisation=False, mixedAttention=False):
        in_pic = batch[0][0][batch.shape[2] // 3]
        tf.summary.image('original_slice', tf.expand_dims(tf.expand_dims(in_pic, axis=0), axis=3), max_outputs=1)

        conv1 = self.n_convs(batch, [(64, 2), (64, 1), (64, 1), (64, 1)])
        # cconv1 = self.n_convs(conv1, [(128, 2), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1)])
        conv2 = self.n_convs(conv1, [(128, 2), (128, 1), (128, 1), (128, 1)])
        conv3 = self.n_convs(conv2, [(256, 2), (256, 1), (256, 1), (256, 1)])

        logits = self.classify(conv3)

        if attention:
            compatability_s1, mixed_loss_s1 = GridAttentionBlock(mixedAttention, self.batch_hard_soft_maps, conv3)(conv1)
            compatability_s2, mixed_loss_s2 = GridAttentionBlock(mixedAttention, self.batch_hard_soft_maps, conv3)(conv2)
            attention_logits_s1 = self.classify(compatability_s1)
            attention_logits_s2 = self.classify(compatability_s2)

            # logits = tf.reduce_mean(tf.stack([logits, attention_logits], 0), 0)
            logits = self.max_prediction(logits, attention_logits_s1, attention_logits_s2)
            # logits = self.fc_predictions(logits, attention_logits_s1, attention_logits_s2)

        if localisation:
            flat = tf.layers.flatten(conv3)
            logits = self.dense_layer(flat, 3, act_f=tf.tanh, dropout=False)

        return logits, mixed_loss_s1 + mixed_loss_s2

    def __init__(self, input_shape, lr, weight_decay, global_step, attention, localisation, mixedAttention):
        super().__init__(input_shape, lr, weight_decay, global_step, localisation, mixedAttention)

        def projection_shortcut(net, out_channels, filter_strides, padding='SAME'):
            return Conv3D(net, out_channels, 1, strides=filter_strides, padding=padding, data_format="channels_first")
        self.p_s = projection_shortcut

        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())
        print('shape', self.batch_features.shape)

        net = tf.expand_dims(self.batch_features, axis=1)
        print(net.shape)

        # net = make_parallel(self.build_long_net, 1, input=net)
        # net = make_parallel(self.build_ti_net, 1, input=net)
        net, mixed_loss = self.build_ti_net(net, attention=attention, localisation=localisation, mixedAttention=mixedAttention)
        print(net.shape)

        self.build(net, mixed_loss)
