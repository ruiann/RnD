from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reader import *
from rnd import RnD
import tensorflow as tf
import random

rate = 0.0001
loop = 1000000
batch_size = 512
log_dir = './log'
model_dir = './model'

data_buckets, label_buckets = read()


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
    return bucket_index, x_feed, label_feed


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
            summary_str, loss_value, _ = sess.run([summary, loss, train_op], feed_dict={x: x_batch, label: label_batch})
            summary_writer.add_summary(summary_str, step)
            print('bucket: {}'.format(bucket_index))
            print('loss: {}'.format(loss_value))

            if step % 10000 == 0 and step != 0:
                checkpoint_file = os.path.join(model_dir, 'model')
                saver.save(sess, checkpoint_file, step)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            sess.run(update_global_step)
            step = global_step.eval()

        summary_writer.close()


if __name__ == '__main__':
    train_recognizer()
