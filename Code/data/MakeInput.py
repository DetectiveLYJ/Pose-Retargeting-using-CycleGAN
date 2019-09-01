import os
import json
import numpy as np


def make_dataset(dir):
    '''
    1 load the data
    2 save data as a tensor
    3 compute the mean and std of data
    4 norminaze the data
    # 5 add device to data
    6 return
    '''
    pose_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            pose_paths.append(path)
    paths = sorted(pose_paths)
    return paths



