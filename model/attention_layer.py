import tensorflow as tf
import numpy as np

def image_summary(name, a_m, stretch=True):
    img_sign = tf.divide(tf.reduce_sum(a_m), tf.abs(tf.reduce_sum(a_m)))
    im = tf.expand_dims(tf.expand_dims(a_m, axis=0), axis=3) * img_sign
    if stretch:
        im = tf.divide(tf.subtract(im, tf.reduce_min(im)), tf.reduce_max(im)) * 255
    tf.summary.image(name, im, max_outputs=1)

class GridAttentionBlock():
    def __init__(self, mixedAttention, batch_hard_soft_maps, gate):
        self.mixedAttention = mixedAttention
        self.batch_hard_soft_maps = batch_hard_soft_maps
        self.g = gate

    def __call__(self, f):
        in_channels, gating_channels = f.shape[1], self.g.shape[1]
        inter_channels = in_channels // 2

        mapped_f = tf.layers.conv3d(f, inter_channels, 1, strides=1, padding='SAME', data_format="channels_first", use_bias=False)
        mapped_g = tf.layers.conv3d(self.g, inter_channels, 1, strides=1, padding='SAME', data_format="channels_first", use_bias=True)

        scale = [f.shape[i] // self.g.shape[i] for i in range(2, 5)]
        upsampled_mapped_g = tf.keras.layers.UpSampling3D(scale, data_format='channels_first')(mapped_g)
        print('mapped_f', mapped_f.shape)
        print('upsampled_g', upsampled_mapped_g.shape)
        combined = tf.nn.relu(tf.add(mapped_f, upsampled_mapped_g))
        attention = tf.layers.conv3d(combined, inter_channels, 1, strides=1, padding='SAME', data_format="channels_first")

        print(attention.shape)
        shifted_attention = tf.subtract(attention, tf.reduce_min(attention))
        normalised_attention = tf.divide(shifted_attention, tf.reduce_sum(shifted_attention))
        print(normalised_attention.shape)
        # normalised_attention = tf.nn.softmax(attention)

        a_m = normalised_attention[0][20][1]
        tf.summary.scalar('max_div_attention', tf.divide(tf.reduce_max(a_m), tf.reduce_min(a_m)))

        attended_attention = tf.multiply(normalised_attention, mapped_f)

        image_summary('f', f[0][20][f.shape[2] // 3])
        image_summary('mapped_f', mapped_f[0][20][mapped_f.shape[2] // 3])
        image_summary('attention map', a_m)
        image_summary('attended attention map', attended_attention[0][20][attended_attention.shape[2] // 3])

        # Hard-soft attention
        if self.mixedAttention:
            self.batch_hard_soft_maps = tf.expand_dims(self.batch_hard_soft_maps, axis=1)
            scale = [(self.batch_hard_soft_maps.shape[i] // normalised_attention.shape[i]).value for i in range(2, 5)]
            maps = self.batch_hard_soft_maps
            print(scale)
            if scale[0] != 1:
                maps = tf.layers.max_pooling3d(maps, scale, scale, data_format="channels_first")
            image_summary('mixed_attention_map', maps[0][0][f.shape[2] // 3])
            print('maps', maps.shape)
            print('normed', normalised_attention.shape)

            mixed_loss = tf.reduce_mean(tf.square(normalised_attention - maps))
            print('Mixed attention configured')

        return attended_attention, mixed_loss
