from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BidirectionalLSTM:
    def __init__(self, name, lstm_size, data_type=tf.float32):
        self.data_type = data_type
        self.name = name
        with tf.variable_scope(self.name):
            self.forward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.backward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    def run(self, data, reuse=False, time_major=False):
        with tf.variable_scope(self.name):
            with tf.variable_scope("ForwardLSTM", reuse=reuse) as scope:
                time_axis = 0 if time_major else 1
                forward_output, state = tf.nn.dynamic_rnn(self.forward_lstm, data, dtype=self.data_type, time_major=time_major, scope=scope)
                forward_output = tf.reduce_mean(forward_output, time_axis)

            with tf.variable_scope("BackwardLSTM", reuse=reuse) as scope:
                time_axis = 0 if time_major else 1
                data = tf.reverse(data, axis=[time_axis])
                backward_output, state = tf.nn.dynamic_rnn(self.backward_lstm, data, dtype=self.data_type, time_major=time_major, scope=scope)
                backward_output = tf.reduce_mean(backward_output, time_axis)

            tf.summary.histogram('forward_lstm_output', forward_output)
            tf.summary.histogram('backward_lstm_output', backward_output)

        return tf.add(forward_output, backward_output, 'feature')
