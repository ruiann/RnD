from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layer import *
from tensorflow.contrib import rnn


class RnD:
    def __init__(self, batch_size=32, data_type=tf.float32, label=3755):
        self.batch_size = batch_size
        self.data_type = data_type
        self.label = label

    def rnn_encode(self, x, reuse=False, time_major=False, pooling=False):
        rnn_size = [100, 500]
        time_axis = 0 if time_major else 1
        with tf.variable_scope('encoder', reuse=reuse):
            forward_cell = rnn.MultiRNNCell([rnn.GRUCell(rnn_size[i]) for i in range(len(rnn_size))])
            backward_cell = rnn.MultiRNNCell([rnn.GRUCell(rnn_size[i]) for i in range(len(rnn_size))])
            output, state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, x, dtype=self.data_type, time_major=time_major)
            forward_output, backward_output = output

            if pooling == 'mean':
                forward_output = tf.reduce_mean(forward_output, time_axis)
                backward_output = tf.reduce_mean(backward_output, time_axis)
            else:
                forward_output = forward_output[-1, :, :] if time_major else forward_output[:, -1, :]
                backward_output = backward_output[-1, :, :] if time_major else backward_output[:, -1, :]

            tf.summary.histogram('forward_rnn_output', forward_output)
            tf.summary.histogram('backward_rnn_output', backward_output)

            code = (forward_output + backward_output) / 2
            with tf.variable_scope('encoder_classification'):
                logit = full_connection_layer(code, self.label)

            tf.summary.histogram('classification', logit)

        self.encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logit
