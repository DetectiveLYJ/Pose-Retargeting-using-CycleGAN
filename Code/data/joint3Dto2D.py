import os
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw3Dpose(joints):      #Painting 3D joints
    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
               [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        point1_index = limb[0]
        point2_index = limb[1]
        point1 = joints[point1_index,:]
        point2 = joints[point2_index,:]
        X = [point1[0], point2[0]]
        Y = [point1[1], point2[1]]
        Z = [point1[2], point2[2]]
        ax.plot3D(X,Y,Z)

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
    plt.show()
    pass


def draw2Dpose(joints):      #Painting 2D joints
    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
               [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]

    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        point1_index = limb[0]
        point2_index = limb[1]
        point1 = joints[point1_index,:]
        point2 = joints[point2_index,:]
        X = [point1[0], point2[0]]
        Y = [point1[1], point2[1]]

        plt.plot(X,Y)

    plt.show()
    pass


def main(src_dir,out_dir,char_names):

    for char in char_names:
        i = 1
        char_dir=os.path.join(src_dir, char)
        file_names = os.listdir(char_dir)
        file_names.sort()

        for file in file_names:
            path=os.path.join(char_dir, file)          #The path of json file

            with open(path,'r') as pose:
                json_f=json.load(pose)
                joints3D=np.array(json_f['pose_keypoints_3d']).reshape((15,3))

                joints2D=joints3D[:,[0,2]]
                json_out = {"pose_keypoints_2d": joints2D.flatten().tolist()}

                save_dir = os.path.join(out_dir,char)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, '%04d.json' % i)
                with open(save_path, 'w') as f:
                    json.dump(json_out, f)
                i = i + 1

if __name__ == '__main__':
    main()



