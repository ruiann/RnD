from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import json

data_dir = './OLHWDB1.1trn_pot_decode'
test_dir = './OLHWDB1.1tst_pot_decode'
data_target_dir = './data'
test_target_dir = './test'
bucket_gap = 10
bucket_num = 10


def pad(length, data):
    pad_length = length - len(data)
    if pad_length > 0:
        eos = np.array([[0, 0, 0, 0, 1]] * pad_length, np.float32)
        data = np.concatenate((data, eos), axis=0)
    else:
        data = data[0: length]
    return np.array(data, np.float32)


def norm(sequence, std=None):
    sequence = np.array(sequence, dtype=np.float32)
    mean = sequence.mean()
    std = std or sequence.std()
    sequence = (sequence - mean) / std
    return sequence, std


def read_file(path):
    new_data = []
    data = np.loadtxt(path)
    data[:, 0], std = norm(data[:, 0])
    data[:, 1], _ = norm(data[:, 1], std)
    prev_x = 0
    prev_y = 0
    for point in data:
        new_data.append([point[0] - prev_x, point[1] - prev_y, point[2], point[3], point[4]])
        prev_x = point[0]
        prev_y = point[1]
    bucket_index = min(int(len(data) / bucket_gap), bucket_num - 1)
    length = bucket_gap * (bucket_index + 1)
    data = pad(length, new_data)
    return data.tolist(), bucket_index


def init_bucket():
    data = []
    for i in range(bucket_num):
        data.append([])
    return data


def init_data(dir, target_dir):
    data_buckets = init_bucket()
    label_buckets = init_bucket()
    index = 0
    for label in os.listdir(dir):
        label_dir = os.path.join(dir, label)
        for writer in os.listdir(label_dir):
            data, bucket_index = read_file(os.path.join(label_dir, writer))
            label_buckets[bucket_index].append(index)
            data_buckets[bucket_index].append(data)
        index = index + 1
    for i in range(bucket_num):
        with open('{}/data_bucket_{}.json'.format(target_dir, i), 'w') as json_file:
            json_file.write(json.dumps(data_buckets[i]))
        with open('{}/label_bucket_{}.json'.format(target_dir, i), 'w') as json_file:
            json_file.write(json.dumps(label_buckets[i]))
    return data_buckets, label_buckets, index


def read(dir):
    data_buckets = []
    label_buckets = []
    for i in range(bucket_num):
        with open('{}/data_bucket_{}.json'.format(dir, i)) as json_file:
            data = json.load(json_file)
            data_buckets.append(data)
        with open('{}/label_bucket_{}.json'.format(dir, i)) as json_file:
            label = json.load(json_file)
            label_buckets.append(label)
    return data_buckets, label_buckets


def read_data():
    return read(data_target_dir)


def read_test():
    return read(test_target_dir)


if __name__ == '__main__':
    init_data(data_dir, data_target_dir)
    init_data(test_dir, test_target_dir)
