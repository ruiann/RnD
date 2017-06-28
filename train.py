from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reader import *
from rnd import RnD
import tensorflow as tf
import random

rate = 0.0001
loop = 1000000
batch_size = 64
log_dir = './log'
model_dir = './model'

data_buckets, label_buckets = read()


def feed_dict(batch_size):
    s_feed = []
    label_feed = []
    bucket_index = random.randint(0, len(data_buckets) - 1)
    data_bucket = data_buckets[bucket_index]
    label_bucket = data_buckets[bucket_index]
    for i in range(batch_size):
        index = random.randint(0, len(data_bucket) - 1)
        data = data_bucket[index]
        label = label_bucket[index]
        s_feed.append(data)
        label_feed.append(label)
    return bucket_index, s_feed, label_feed


def train():
    x = tf.placeholder(tf.float32, [batch_size, None, 5])
    label = tf.placeholder(tf.float32, [batch_size])
    model = RnD(batch_size=batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)

        logits = model.rnn_encode(x, batch_size)
        classification = tf.to_int32(tf.arg_max(tf.nn.softmax(logits), dimension=1))
        differ = label - classification
        tf.summary.histogram('classification difference', differ)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
        tf.summary.scalar('classifier loss', loss)
        train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        step = global_step.eval()
        while step < loop:
            print('step: %d' % step)
            bucket_index, x_batch, label_batch = feed_dict(batch_size)
            summary_str, loss = sess.run([summary, train_op], feed_dict={x: x_batch, label: label_batch})
            summary_writer.add_summary(summary_str, step)

            if global_step.eval() % 20 == 0:
                checkpoint_file = os.path.join(model_dir, 'model.latest')
                saver.save(sess, checkpoint_file)

            if global_step.eval() % 100 == 0:
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            sess.run(update_global_step)

        summary_writer.close()


if __name__ == '__main__':
    train()