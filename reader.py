from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

data_dir = './OLHWDB1.1trn_pot_decode'
target_dir = './data'
bucket_gap = 10
bucket_num = 10


def pad(length, data):
    pad = length - len(data)
    if pad > 0:
        eos = np.array([[0, 0, 0, 0, 1]] * pad, np.float32)
        data = np.concatenate((data, eos), axis=0)
    else:
        data = data[0: length]
    return data


def norm(sequence, std=None):
    sequence = np.array(sequence, dtype=np.float32)
    mean = sequence.mean()
    std = std or sequence.std()
    sequence = 100 * (sequence - mean) / std
    return sequence, std


def read_file(path):
    data = np.loadtxt(path)
    data[:, 0], std = norm(data[:, 0])
    data[:, 1], _ = norm(data[:, 1], std)
    bucket_index = min(int(len(data) / bucket_gap), bucket_num - 1)
    length = bucket_gap * (bucket_index + 1)
    data = pad(length, data)
    return data, bucket_index


def init_bucket():
    data = []
    for i in range(bucket_num):
        data.append([])
    return data


def init():
    buckets = init_bucket()
    index = 0
    for label in os.listdir(data_dir):
        print(label)
        label_dir = os.path.join(data_dir, label)
        for writer in os.listdir(label_dir):
            data, bucket_index = read_file(os.path.join(label_dir, writer))
            data = {'data': data, 'label': index}
            buckets[bucket_index].append(data)
        index = index + 1
    for i in range(bucket_num):
        np.savetxt(os.path.join(target_dir, '{}.txt'.format(i)), buckets[i])
    return buckets, index


if __name__ == '__main__':
    data, class_num = init()
    print(class_num)
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))