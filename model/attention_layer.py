import tensorflow as tf
import numpy as np

class GridAttentionBlock():
    # f: features, g: gating signal
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        self.in_channels = in_channels
        self.gating_channels = gating_channels

        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

    def __call__(self, f, g):
        print('f', f.shape)
        print('g', g.shape)

        f_im = f[0][40][2]
        print(f_im.shape)
        tf.summary.image('f', tf.expand_dims(tf.expand_dims(f_im, axis=0), axis=3), max_outputs=10)
        g_im = g[0][40][2]
        tf.summary.image('g', tf.expand_dims(tf.expand_dims(g_im, axis=0), axis=3), max_outputs=10)

        mapped_f = tf.layers.conv3d(f, self.inter_channels, 1, strides=1, padding='SAME', data_format="channels_first")
        scale = [f.shape[i] // g.shape[i] for i in range(2, 5)]

        mapped_g = tf.layers.conv3d(g, self.inter_channels, 1, strides=1, padding='SAME', data_format="channels_first")
        upsampled_mapped_g = tf.keras.layers.UpSampling3D(scale, data_format='channels_first')(mapped_g)

        combined = tf.nn.relu(tf.keras.layers.Add()([mapped_f, upsampled_mapped_g]))
        attention = tf.layers.conv3d(combined, self.inter_channels, 1, strides=1, padding='SAME', data_format="channels_first")

        shifted_attention = tf.math.subtract(attention, tf.math.reduce_min(attention))
        normalised_attention = tf.math.divide(shifted_attention, tf.math.reduce_sum(shifted_attention))

        a_m = normalised_attention[0][40][2]
        im = tf.expand_dims(tf.expand_dims(a_m, axis=0), axis=3) * 1e8
        tf.summary.scalar('max_diff_attention', tf.math.reduce_max(im) - tf.math.reduce_min(im))
        im = tf.divide(tf.subtract(im, tf.reduce_min(im)), tf.reduce_max(im)) * 255
        tf.summary.image('attention map', im, max_outputs=10)

        attended_attention = tf.math.multiply(normalised_attention, mapped_f)
        a_m = attended_attention[0][40][2]
        tf.summary.image('attended attention map', tf.expand_dims(tf.expand_dims(a_m, axis=0), axis=3), max_outputs=10)
        return attended_attention
