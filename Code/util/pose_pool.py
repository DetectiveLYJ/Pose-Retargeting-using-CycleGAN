import random
import torch


class PosePool():
    """This class implements an pose buffer that stores previously generated poses.

    This buffer enables us to update discriminators using a history of generated poses
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the PosePool class

        Parameters:
            pool_size (int) -- the size of pose buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_poses = 0
            self.poses = []

    def query(self, poses):
        """Return an pose from the pool.

        Parameters:
            poses: the latest generated poses from the generator

        Returns poses from the buffer.

        By 50/100, the buffer will return input poses.
        By 50/100, the buffer will return poses previously stored in the buffer,
        and insert the current poses to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return poses
        return_poses = []
        for pose in poses:
            pose = torch.unsqueeze(pose.data, 0)
            if self.num_poses < self.pool_size:   # if the buffer is not full; keep inserting current poses to the buffer
                self.num_poses = self.num_poses + 1
                self.poses.append(pose)
                return_poses.append(pose)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored pose, and insert the current pose into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.poses[random_id].clone()
                    self.poses[random_id] = pose
                    return_poses.append(tmp)
                else:       # by another 50% chance, the buffer will return the current pose
                    return_poses.append(pose)
        return_poses = torch.cat(return_poses, 0)   # collect all the poses and return
        return return_poses
