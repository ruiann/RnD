from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imp import reload
import struct
import sys
import numpy as np
import os

reload(sys)
sys.setdefaultencoding('utf-8')


def decode(path, target_path, writer):
    path = '{}{}-c.pot'.format(path, writer)
    f = open(path, 'rb')
    while f.read(2):
        tag_code_1 = f.read(1)
        tag_code_0 = f.read(1)
        tag_code = tag_code_0 + tag_code_1
        tag_code = tag_code.decode('gb2312')
        f.read(2)
        print(tag_code)
        stroke_num = struct.unpack('H', f.read(2))[0]
        print(stroke_num)

        strokes = []
        state = []
        for i in range(stroke_num):
            end = False
            stroke = []
            s = []
            while not end:
                x = struct.unpack('h', f.read(2))[0]
                y = struct.unpack('h', f.read(2))[0]
                if x == -1 and y == 0:
                    end = True
                    s[-1] = [0, 1, 0]
                else:
                    stroke.append([x, y])
                    s.append([1, 0, 0])
            strokes.extend(stroke)
            state.extend(s)
        state[-1] = [0, 0, 1]
        sample = np.concatenate((np.array(strokes), np.array(state)), 1)

        dir_path = '{}{}'.format(target_path, tag_code)
        filename = '{}{}/{}.txt'.format(target_path, tag_code, writer)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.savetxt(filename, sample)
        f.read(4)
    f.close()

if __name__ == '__main__':
    for w in range(1001, 1241):
        decode('./OLHWDB1.1trn_pot/', './OLHWDB1.1trn_pot_decode/', w)