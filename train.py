from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reader import *
from rnd import *
import tensorflow as tf
import random

rate = 0.0001
loop = 500000
batch_size = 512
r_log_dir = './r_log'
r_model_dir = './r_model'
d_log_dir = './d_log'
d_model_dir = './d_model'

data_buckets, label_buckets = read_data()


def feed_dict(batch_size):
    x_feed = []
    label_feed = []
    bucket_index = random.randint(0, len(data_buckets) - 1)
    data_bucket = data_buckets[bucket_index]
    label_bucket = label_buckets[bucket_index]
    for i in range(batch_size):
        index = random.randint(0, len(data_bucket) - 1)
        data = data_bucket[index]
        label = label_bucket[index]
        x_feed.append(data)
        label_feed.append(label)
    return bucket_index, np.array(x_feed, np.float32), np.array(label_feed, np.int32)


def train_recognizer():
    x = tf.placeholder(tf.float32, [batch_size, None, 5])
    label = tf.placeholder(tf.int32, [batch_size])
    model = RnD(batch_size=batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)

        logits = model.classification(model.rnn_encode(x))
        classification = tf.to_int32(tf.arg_max(tf.nn.softmax(logits), dimension=1))
        differ = label - classification
        tf.summary.histogram('classification difference', differ)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
        tf.summary.scalar('classifier loss', loss)
        train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(r_model_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(r_log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        step = global_step.eval()
        while step < loop:
            print('step: %d' % step)
            bucket_index, x_batch, label_batch = feed_dict(batch_size)
            summary_str, loss_value, _ = sess.run([summary, loss, train_op], feed_dict={x: x_batch, label: label_batch})
            summary_writer.add_summary(summary_str, step)
            print('bucket: {}'.format(bucket_index))
            print('loss: {}'.format(loss_value))

            if step % 10000 == 0 and step != 0:
                checkpoint_file = os.path.join(r_model_dir, 'model')
                saver.save(sess, checkpoint_file, step)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            sess.run(update_global_step)
            step = global_step.eval()

        summary_writer.close()


def train_drawer():
    model = RnD(batch_size=batch_size)

    x = tf.placeholder(tf.float32, [batch_size, None, 5], 'x')
    coding = model.rnn_encode(x, training=False)

    code = tf.placeholder(tf.float32, [batch_size, 500], 'code')
    state = tf.placeholder(tf.float32, [batch_size, 500], 'state')
    d_prev = tf.placeholder(tf.float32, [batch_size, 2], 'prev_d')
    s_prev = tf.placeholder(tf.float32, [batch_size, 3], 'prev_s')
    d = tf.placeholder(tf.float32, [batch_size, 2], 'd')
    s = tf.placeholder(tf.float32, [batch_size, 3], 's')

    out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y, status, rnn_state = model.rnn_decode_step(code, d_prev, s_prev, [state])
    loss_d = get_loss_func_d(d, out_pi, out_sigma_x, out_mu_x, out_sigma_y, out_mu_y)
    loss_s = get_loss_func_s(status, s)
    loss = (loss_d + loss_s) / 2
    tf.summary.histogram('d_loss', loss_d)
    tf.summary.histogram('s_loss', loss_s)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, var_list=model.decoder_variables)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)

        sess.run(tf.global_variables_initializer())
        r_saver = tf.train.Saver(model.encoder_variables)
        d_saver = tf.train.Saver(model.decoder_variables)
        r_ckpt = tf.train.get_checkpoint_state(r_model_dir)
        d_ckpt = tf.train.get_checkpoint_state(d_model_dir)
        if r_ckpt:
            r_saver.restore(sess, r_ckpt.model_checkpoint_path)
        if d_ckpt:
            d_saver.restore(sess, d_ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(d_log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        step = global_step.eval()
        while step < loop:
            print('step: %d' % step)
            bucket_index, x_batch, _ = feed_dict(batch_size)
            print('bucket: {}'.format(bucket_index))
            code_value = sess.run(coding, feed_dict={x: x_batch})
            rnn_state_value = np.zeros([batch_size, 500], np.float32)
            for i in range(bucket_gap * bucket_index + bucket_gap):
                if i == 0:
                    prev_d = np.zeros([batch_size, 2], np.float32)
                    prev_s = np.zeros([batch_size, 3], np.float32)
                else:
                    prev_d = x_batch[:, i - 1, 0: 2]
                    prev_s = x_batch[:, i - 1, 2: 5]
                d_value = x_batch[:, i, 0: 2]
                s_value = x_batch[:, i, 2: 5]
                [rnn_state_value], summary_str, loss_value, _ = sess.run([rnn_state, summary, loss, train_op], feed_dict={
                    code: code_value,
                    state: rnn_state_value,
                    d_prev: prev_d,
                    s_prev: prev_s,
                    d: d_value,
                    s: s_value
                })
                summary_writer.add_summary(summary_str, step)
                print('loss: {}'.format(loss_value))

            if step % 1000 == 0 and step != 0:
                checkpoint_file = os.path.join(d_model_dir, 'model')
                d_saver.save(sess, checkpoint_file, step)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            sess.run(update_global_step)
            step = global_step.eval()

        summary_writer.close()


if __name__ == '__main__':
    # train_recognizer()
    train_drawer()
