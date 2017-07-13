from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layer import *
from tensorflow.contrib import rnn
import math


def normal(y, mu, sigma):
    result = tf.subtract(y, mu)
    result = tf.multiply(result, tf.reciprocal(sigma))
    result = - tf.square(result) / 2
    return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) / math.sqrt(2 * math.pi)


def get_loss_func_d(d, out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y):
    x, y = tf.split(d, num_or_size_splits=2, axis=1)
    p_x = normal(x, out_mu_x, out_sigma_x)
    p_y = normal(y, out_mu_y, out_sigma_y)
    p = tf.multiply(p_x, p_y)
    tf.summary.histogram('gmm_prob', p)
    p = tf.multiply(p, out_pi)
    result = tf.reduce_sum(p, 1, keep_dims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)


def get_loss_func_s(logits, targets):
    pos_weight = tf.constant([1, 5, 100], tf.float32)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=logits, pos_weight=pos_weight))


class RnD:
    def __init__(self, batch_size=32, k=20, data_type=tf.float32, label=3755, encoder_rnn_size=[100, 500], decoder_rnn_size=[500]):
        self.batch_size = batch_size
        self.data_type = data_type
        self.label = label
        self.k = k
        self.encoder_rnn_size = encoder_rnn_size
        self.decoder_rnn_size = decoder_rnn_size

        self.encoder_variables = []
        self.decoder_variables = []

    def rnn_encode(self, x, reuse=False, time_major=False, pooling='mean', training=True):
        time_axis = 0 if time_major else 1
        with tf.variable_scope('encoder', reuse=reuse):
            forward_cell = rnn.MultiRNNCell([rnn.GRUCell(self.encoder_rnn_size[i]) for i in range(len(self.encoder_rnn_size))])
            backward_cell = rnn.MultiRNNCell([rnn.GRUCell(self.encoder_rnn_size[i]) for i in range(len(self.encoder_rnn_size))])
            forward_output, _ = tf.nn.dynamic_rnn(forward_cell, x, dtype=self.data_type, time_major=time_major, scope="forward")
            x = tf.reverse(x, axis=[time_axis])
            backward_output, _ = tf.nn.dynamic_rnn(backward_cell, x, dtype=self.data_type, time_major=time_major, scope="backward")

            if pooling == 'mean':
                forward_output = tf.reduce_mean(forward_output, time_axis)
                backward_output = tf.reduce_mean(backward_output, time_axis)
            else:
                forward_output = forward_output[-1, :, :] if time_major else forward_output[:, -1, :]
                backward_output = backward_output[-1, :, :] if time_major else backward_output[:, -1, :]

            if training:
                tf.summary.histogram('forward_rnn_output', forward_output)
                tf.summary.histogram('backward_rnn_output', backward_output)

            code = (forward_output + backward_output) / 2

        self.encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        return code

    def classification(self, code, training=True):
        with tf.variable_scope('encoder_classification'):
            with tf.variable_scope('layer1'):
                code = full_connection_layer(code, 200)
                code = tf.nn.relu(code)
            with tf.variable_scope('layer2'):
                code = full_connection_layer(code, self.label)

        if training:
            tf.summary.histogram('classification', code)
        return code

    def rnn_decode_step(self, code, d, s, state, reuse=False, training=True):
        stddev = 0.5
        with tf.variable_scope('decoder', reuse=reuse):
            Wd = tf.Variable(tf.random_normal([2, 500], stddev=stddev, dtype=tf.float32))
            bd = tf.Variable(tf.random_normal([500], stddev=stddev, dtype=tf.float32))
            Ws = tf.Variable(tf.random_normal([3, 500], stddev=stddev, dtype=tf.float32))
            bs = tf.Variable(tf.random_normal([500], stddev=stddev, dtype=tf.float32))
            x = (tf.nn.tanh(tf.matmul(d, Wd) + bd) + tf.nn.tanh(tf.matmul(s, Ws) + bs) + code) / 2

            cell = rnn.MultiRNNCell([rnn.GRUCell(self.decoder_rnn_size[i]) for i in range(len(self.decoder_rnn_size))])

            output, state = cell(x, state)

            Wst = tf.Variable(tf.random_normal([self.decoder_rnn_size[-1], 3], stddev=stddev, dtype=tf.float32))
            bst = tf.Variable(tf.random_normal([3], stddev=stddev, dtype=tf.float32))
            status = tf.matmul(output, Wst) + bst

            Wh = tf.Variable(tf.random_normal([self.decoder_rnn_size[-1], 5 * self.k], stddev=stddev, dtype=tf.float32))
            bh = tf.Variable(tf.random_normal([5 * self.k], stddev=stddev, dtype=tf.float32))
            gmm = tf.nn.tanh(tf.matmul(output, Wh) + bh)
            out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y = tf.split(gmm, num_or_size_splits=5, axis=1)

            max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
            out_pi = tf.subtract(out_pi, max_pi)
            out_pi = tf.exp(out_pi)
            normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
            out_pi = tf.multiply(normalize_pi, out_pi)

            out_sigma_x = tf.exp(out_sigma_x)
            out_sigma_y = tf.exp(out_sigma_y)
        self.decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

        if training:
            tf.summary.histogram('rnn_output', output)
            tf.summary.histogram('gmm_pi', out_pi)
            tf.summary.histogram('gmm_mu_x', out_mu_x)
            tf.summary.histogram('gmm_sigma_x', out_sigma_x)
            tf.summary.histogram('gmm_mu_y', out_mu_y)
            tf.summary.histogram('gmm_sigma_y', out_sigma_y)
            tf.summary.histogram('predict_s', status)

        return out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y, status, state
