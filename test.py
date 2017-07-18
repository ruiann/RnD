from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reader import *
from rnd import *
import tensorflow as tf
import random

rate = 0.001
loop = 500000
batch_size = 512
r_log_dir = './r_log'
r_model_dir = './r_model'
d_log_dir = './d_log'
d_model_dir = './d_model'

data_buckets, label_buckets = read_test()


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


def compare(label, infer):
    right = 0
    mistake = 0
    for i in range(len(label)):
        if label[i] == infer[i]:
            right = right + 1
        else:
            mistake = mistake + 1
    return right, mistake


def test_recognizer():
    x = tf.placeholder(tf.float32, [batch_size, None, 5])
    model = RnD(batch_size=batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        logits = model.classification(model.rnn_encode(x))
        inferred = tf.arg_max(logits, 1)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(r_model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        step = 0
        while step < loop:
            bucket_index, x_batch, label_batch = feed_dict(batch_size)
            classification = sess.run(inferred, feed_dict={x: x_batch})
            right, mistake = compare(label_batch, classification)
            print('bucket: {}'.format(bucket_index))
            print('right: {} mistake: {}'.format(right, mistake))

            step = step + 1


if __name__ == '__main__':
    test_recognizer()
