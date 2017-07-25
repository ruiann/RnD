from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layer import *
from tensorflow.contrib import rnn
import math


def sample(pi, mu_x, mu_y, name):
    x = tf.multiply(mu_x, pi)
    y = tf.multiply(mu_y, pi)
    return tf.concat([x, y], 1, name=name)


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
    p = tf.multiply(p, out_pi)
    result = tf.reduce_sum(p, 1, keep_dims=True)
    result = -tf.log(result)
    return result


def get_loss_func_s(targets, logits):
    pos_weight = tf.constant([1, 5, 100], tf.float32)
    return tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=logits, pos_weight=pos_weight)


class RnD:
    def __init__(self, batch_size=32, k=30, data_type=tf.float32, label=3755, encoder_rnn_size=[100, 500], decoder_rnn_size=[1000], dim=300):
        self.batch_size = batch_size
        self.data_type = data_type
        self.label = label
        self.k = k
        self.encoder_rnn_size = encoder_rnn_size
        self.decoder_rnn_size = decoder_rnn_size
        self.dim = dim

        self.encoder_variables = []
        self.decoder_variables = []
        self.init_decoder()

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

    def init_decoder(self, reuse=False):
        stddev = 0.5
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope('coding', reuse=reuse):
                self.Wd = tf.Variable(tf.random_normal([2, self.dim], stddev=stddev, dtype=tf.float32), name='Wd')
                self.bd = tf.Variable(tf.random_normal([self.dim], stddev=stddev, dtype=tf.float32), name='bd')
                self.Ws = tf.Variable(tf.random_normal([3, self.dim], stddev=stddev, dtype=tf.float32), name='Ws')
                self.bs = tf.Variable(tf.random_normal([self.dim], stddev=stddev, dtype=tf.float32), name='bs')
                self.Wc = tf.Variable(tf.random_normal([500, self.dim], stddev=stddev, dtype=tf.float32), name='Wc')
                self.bc = tf.Variable(tf.random_normal([self.dim], stddev=stddev, dtype=tf.float32), name='bc')

                self.Wi = tf.Variable(tf.random_normal([self.dim, self.decoder_rnn_size[-1]], stddev=stddev, dtype=tf.float32), name='Wi')
                self.bi = tf.Variable(tf.random_normal([self.decoder_rnn_size[-1]], stddev=stddev, dtype=tf.float32), name='bi')

                self.cell = rnn.MultiRNNCell([rnn.GRUCell(size) for size in self.decoder_rnn_size])

            with tf.variable_scope('output', reuse=reuse):
                self.Wst = tf.Variable(tf.random_normal([self.decoder_rnn_size[-1], 3], stddev=stddev, dtype=tf.float32), name='Wst')
                self.bst = tf.Variable(tf.random_normal([3], stddev=stddev, dtype=tf.float32), name='bst')

                self.Wh = tf.Variable(tf.random_normal([self.decoder_rnn_size[-1], 5 * self.k], stddev=stddev, dtype=tf.float32), name='Wh')
                self.bh = tf.Variable(tf.random_normal([5 * self.k], stddev=stddev, dtype=tf.float32), name='bh')

        self.decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def rnn_decode_step(self, code, d, s, hidden, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            x = tf.nn.relu(tf.matmul(d, self.Wd) + self.bd) + tf.nn.relu(tf.matmul(s, self.Ws) + self.bs) + tf.nn.relu(tf.matmul(code, self.Wc) + self.bc)
            x = tf.nn.relu(tf.matmul(x, self.Wi) + self.bi)
            output, hidden = self.cell(x, hidden)

            state = tf.matmul(output, self.Wst) + self.bst
            gmm = tf.nn.tanh(tf.matmul(output, self.Wh) + self.bh)
            out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y = tf.split(gmm, num_or_size_splits=5, axis=1)

            max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
            out_pi = tf.subtract(out_pi, max_pi)
            out_pi = tf.exp(out_pi)
            normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
            out_pi = tf.multiply(normalize_pi, out_pi, name='gmm_pi')

            out_sigma_x = tf.exp(out_sigma_x, name='gmm_sigma_x')
            out_sigma_y = tf.exp(out_sigma_y, name='gmm_sigma_y')

        return out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y, state, hidden

    def train(self, x):
        length = tf.shape(x)[1]
        coding = self.rnn_encode(x, training=False)

        i = tf.constant(0)
        hidden = tuple([tf.zeros([self.batch_size, size], tf.float32) for size in self.decoder_rnn_size])
        d_loss = tf.Variable(0, dtype=tf.float32)
        s_loss = tf.Variable(0, dtype=tf.float32)
        default_prev_d = tf.zeros([self.batch_size, 2], tf.float32)
        default_prev_s = tf.zeros([self.batch_size, 3], tf.float32)

        def cond(i, *_):
            return tf.less(i, length)

        def body(i, hidden, d_loss, s_loss):

            def init_call():
                prev_d = default_prev_d
                prev_s = default_prev_s
                return prev_d, prev_s

            def normal_call():
                prev_d = x[:, i - 1, 0: 2]
                prev_s = x[:, i - 1, 2: 5]
                return prev_d, prev_s

            prev_d, prev_s = tf.cond(tf.equal(i, 0), lambda: init_call(), lambda: normal_call())
            d = x[:, i, 0: 2]
            s = x[:, i, 2: 5]

            out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y, state, out_hidden = self.rnn_decode_step(coding, prev_d, prev_s, hidden)
            loss_d = get_loss_func_d(d, out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y)
            loss_s = get_loss_func_s(s, state)

            d_loss = d_loss + tf.reduce_mean(loss_d, name='frame_loss_d')
            s_loss = s_loss + tf.reduce_mean(loss_s, name='frame_loss_s')
            i = tf.add(i, 1)
            return i, out_hidden, d_loss, s_loss

        _, final_state, final_loss_d, final_loss_s = tf.while_loop(cond=cond, body=body, loop_vars=(i, hidden, d_loss, s_loss))

        final_loss = final_loss_d + final_loss_s
        tf.summary.scalar('loss_d', final_loss_d)
        tf.summary.scalar('loss_s', final_loss_s)
        return final_loss

    def generate(self, x):
        coding = self.rnn_encode(x, training=False)

        i = tf.constant(0)
        hidden = tuple([tf.zeros([self.batch_size, size], tf.float32) for size in self.decoder_rnn_size])
        output = tf.TensorArray(dtype=tf.float32, size=50, dynamic_size=True, tensor_array_name='output')
        prev_d = tf.zeros([self.batch_size, 2], tf.float32)
        prev_s = tf.zeros([self.batch_size, 3], tf.float32)

        def cond(i, prev_d, prev_s, *_):
            return tf.argmax(prev_s, 1) != 2

        def body(i, prev_d, prev_s, hidden, output):
            out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y, state, out_hidden = self.rnn_decode_step(coding,
                                                                                                           prev_d,
                                                                                                           prev_s,
                                                                                                           hidden)

            d = sample(out_pi, out_mu_x, out_mu_y, name='generated_direction')
            s = tf.one_hot(indices=tf.arg_max(state, dimension=-1), depth=3, dtype=tf.float32, name='generated_state')
            output.write(i, tf.concat([d, s], axis=-1, name='generation'))
            i = tf.add(i, 1)
            return i, d, s, out_hidden, output

        i, final_d, final_s, final_hidden, generation = tf.while_loop(cond=cond, body=body, loop_vars=(i, prev_d, prev_s, hidden, output))

        generation = generation.stack(name='sample')
        return generation
