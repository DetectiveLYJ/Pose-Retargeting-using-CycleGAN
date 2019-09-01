"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import json
import numpy as np
import os
import os.path

POSE_EXTENSIONS = [
    '.json'
]


def is_pose_file(filename):
    return any(filename.endswith(extension) for extension in POSE_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    poses = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_pose_file(fname):
                path = os.path.join(root, fname)
                poses.append(path)
    return poses[:min(max_dataset_size, len(poses))]


def default_loader(path):
    pose=[]
    with open(path) as f:
        json_f = json.load(f)
        joints2D = np.array(json_f['pose_keypoints_2d'])  # .reshape((15,2))
        pose.append(joints2D)
    return pose


class PoseFolder(data.Dataset):

    def __init__(self, root, return_paths=False,
                 loader=default_loader):
        poses = make_dataset(root)
        if len(poses) == 0:
            raise(RuntimeError("Found 0 poses in: " + root + "\n"
                               "Supported pose extensions are: " +
                               ",".join(POSE_EXTENSIONS)))

        self.root = root
        self.poses = poses

        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.poses[index]
        pose = self.loader(path)
        if self.return_paths:
            return pose, path
        else:
            return pose

    def __len__(self):
        return len(self.poses)
