from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalLSTM import BidirectionalLSTM


class Encoder:

    def __init__(self, lstm=[100, 500], class_num=10):
        self.bidirectional_LSTM = BidirectionalLSTM('BidirectionalLSTM', lstm, stack=2)
        self.logistic_regression = LogisticRegression('LogisticRegression', lstm[-1], [800, class_num])

    # do classification
    def run(self, data):
        lstm_code = self.lstm(data)
        return self.regression(lstm_code)

    def lstm(self, data):
        return self.bidirectional_LSTM.run(data, reuse=False, time_major=False)

    def regression(self, lstm_code):
        return tf.nn.relu(self.logistic_regression.run(lstm_code)[-1])

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, data, labels):
        logits = self.run(data)
        classification = tf.to_int32(tf.arg_max(tf.nn.softmax(logits), dimension=1))
        differ = labels - classification
        tf.summary.histogram('classification difference', differ)
        loss = self.loss(logits, labels)
        tf.summary.scalar('classifier loss', loss)
        return loss
