import os
import numpy as np
import cv2
import math
import json
from util.util import tensor2np
import matplotlib.pyplot as plt


src_dir = '../../Dataset/2Dpose/Groundtruth/norm'
out_dir = '../../Dataset/2Dpose/Groundtruth/norm_image'

color=[[192,192,192],   # Head = silver
       [128,128,128],   # Neck = gray
       [0,0,128],       # RightArm = navy
       [0,0,255],       # RightForeArm = blue
       [0,255,255],     # RightHand = cyan
       [128,128,0],     # LeftArm = olive
       [0,136,0],       # LeftForeArm = green
       [0,255,0],       # LeftHand = lime
       [0,0,0],         # Hips = black
       [0,128,128],     # RightUpLeg = teal
       [128,0,128],     # RightLeg = purple
       [255,0,255],     # RightFoot = magenta
       [128,0,0],       # LeftUpLeg = maroon
       [255,0,0],       # LeftLeg = red
       [255,255,0]]     # LeftFoot = yellow

H = 256
W = 256

char_names = ['BigVegas', 'LolaB']
maxlength_path = "./data/maxLength.json"


def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]

    return rgb


def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)


def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def bbox_length(joints):
    length = np.max(joints[:, 0]) - np.min(joints[:, 0]), np.max(joints[:, 1]) - np.min(joints[:, 1])
    return length


def bbox_mid(joints):
    bbox=np.min(joints[:,0]),np.max(joints[:,0]),np.min(joints[:,1]),np.max(joints[:,1])
    mid = (bbox[1] + bbox[0])/2 , (bbox[3] + bbox[2])/2
    return mid


def getMaxLength(njoints):          #get MaxLength of all boundingbox
    pose_num=njoints.shape[0]
    all_length=[]

    for i in range(pose_num):
        joints = njoints[i,:,:]
        length = bbox_length(joints)
        all_length.append(length)

    maxLength=np.max(all_length)
    return maxLength


def joints2image_ori(joints_position, colors, transparency=False, H=512, W=512, nr_joints=49, imtype=np.uint8):
    nr_joints = joints_position.shape[0]

    if nr_joints == 49:  # full joints(49): basic(15) + eyes(2) + toes(2) + hands(30)
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16]]

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                         R, M, L, L, L, L, R, R, R,
                         R, R, L] + [L] * 15 + [R] * 15

        colors_limbs = [M, L, R, M, L, L, R,
                        R, L, R, L, L, L, R, R, R,
                        R, R]
    elif nr_joints == 15 or nr_joints == 17:  # basic joints(15) + (eyes(2))
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                         R, M, L, L, L, R, R, R]

        colors_limbs = [M, L, R, M, L, L, R,
                        R, L, R, L, L, R, R]
    else:
        raise ValueError("Only support number of joints be 49 or 17 or 15")

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * 255
    hips = joints_position[8]
    neck = joints_position[1]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5

    head_radius = int(torso_length / 4.5)
    end_effectors_radius = int(torso_length / 15)
    end_effectors_radius = 7
    joints_radius = 7

    cv2.circle(canvas, (int(joints_position[0][0]), int(joints_position[0][1])), head_radius, colors_joints[0],
               thickness=-1)

    for i in range(1, len(colors_joints)):
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
            radius = joints_radius
        cv2.circle(canvas, (int(joints_position[i][0]), int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)

    stickwidth = 2

    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]

        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:, bb[2]:bb[3], :]

    return [canvas.astype(imtype), canvas_cropped.astype(imtype)]


def joints2image(joints_position, colors, H=512, W=512,imtype=np.uint8):
    nr_joints = joints_position.shape[0]

    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]

    canvas = np.ones(shape=(H, W, 3)) * 255     # 3 channels
    hips = joints_position[8]
    neck = joints_position[1]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5
    joint_radius = 3

    for i in range(nr_joints):
        cv2.circle(canvas, (int(joints_position[i][0]), int(joints_position[i][1])), joint_radius, colors[i], thickness=-1)

    stickwidth =1

    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]

        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[8])
        canvas = cv2.addWeighted(canvas, 0.7, cur_canvas, 0.3, 0)

    return canvas.astype(imtype)


def pose2pix(joints_position,imgH):      # from joints positiion to pix position
    with open(maxlength_path,'r') as length:
        maxLength = json.load(length)        # can change

    joints_bias = joints_position - bbox_mid(joints_position)
    joints_bias=np.array(joints_bias)
    tgt_jointpos = (imgH * joints_bias) / maxLength + imgH / 2
    return tgt_jointpos


def pose2img(pose, colors, H=512, W=512):
    pose_numpy = tensor2np(pose)
    pix_numpy = pose2pix(pose_numpy, H)
    image_numpy = joints2image(pix_numpy, colors, H, W)
    image_numpy=cv2.flip(image_numpy, 0)
    return image_numpy


def main():

    for char in char_names:
        char_dir = os.path.join(src_dir, char)
        pose_ids = os.listdir(char_dir)
        pose_ids.sort()

        for pose_id in pose_ids:
            pose_path=os.path.join(char_dir,pose_id)

            with open(pose_path, 'r') as pose:
                joints_position=np.array(json.load(pose)['pose_keypoints_2d']).reshape((15, 2))

            img = joints2image(joints_position,color,H,W)
            img=cv2.flip(img, 0)

            save_dir=os.path.join(out_dir,char)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path=os.path.join(save_dir,pose_id[:-5]+'.jpg')

            cv2.imwrite(save_path,img)
            print(pose_id[:-5] + ' done')


if __name__ == '__main__':
    main()


