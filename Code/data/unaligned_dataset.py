import os.path
from.base_dataset import BaseDataset
from .MakeInput import make_dataset
import torch
import json
import numpy as np
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = make_dataset(self.dir_A)   # load images from '/path/to/data/trainA'
        self.B_paths = make_dataset(self.dir_B)   # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B


    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        with open(A_path) as f:
            json_A = json.load(f)
            jointsA = np.array(json_A['pose_keypoints_2d'])

        with open(B_path) as f:
            json_B = json.load(f)
            jointsB = np.array(json_B['pose_keypoints_2d'])

        A = torch.from_numpy(jointsA).float()
        B = torch.from_numpy(jointsB).float()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):

        return max(self.A_size,self.B_size)
