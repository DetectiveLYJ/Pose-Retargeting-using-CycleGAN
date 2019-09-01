import os
import hashlib
import random
import shutil
import json
import numpy as np
import joint3Dto2D
from pose2image import bbox_mid,getMaxLength


ALL_JSON3D_DIR="../../Dataset/3Djson/allData/"
SAMEDATA3D_DIR="../../Dataset/3Djson/sameData/"
USEDATA3D_DIR="../../Dataset/3Djson/usefullData/"

DATA3D_DIR="../../Dataset/3Djson/Dataset/"
GT3D_DIR="../../Dataset/3Djson/Groundtruth/"

TRAIN3D_DIR="../../Dataset/Data3D/Dataset/train/"
TEST3D_DIR="../../Dataset/Data3D/Dataset/test/"

GT_TRAIN3D_DIR="../../Dataset/Data3D/Groundtruth/train/"
GT_TEST3D_DIR="../../Dataset/Data3D/Groundtruth/test/"

DATA2D_DIR="../../Dataset/2Dpose/Dataset/"
GT2D_DIR="../../Dataset/2Dpose/Groundtruth/"

TRAIN2D_DIR="../../Dataset/Data/Dataset/train/"
TEST2D_DIR="../../Dataset/Data/Dataset/test/"

GT_TRAIN2D_DIR="../../Dataset/Data/Groundtruth/train/"
GT_TEST2D_DIR="../../Dataset/Data/Groundtruth/test/"

maxlength_path="./maxLength.json"

H = 256
W = 256


def selectSameMotion(srcDir,tarDir,character_names):

    charA_dir=os.path.join(srcDir, character_names[0])
    charB_dir=os.path.join(srcDir, character_names[1])
    dataA_dir=os.path.join(tarDir, character_names[0])
    dataB_dir=os.path.join(tarDir, character_names[1])

    if not os.path.exists(dataA_dir):
        os.makedirs(dataA_dir)

    if not os.path.exists(dataB_dir):
        os.makedirs(dataB_dir)

    md5dict={}
    for filename in os.listdir(charA_dir):
        hashvalue=hashlib.md5(filename.encode('utf-8')).hexdigest()
        md5dict[hashvalue]=os.path.join(charA_dir,filename)
    for filename in os.listdir(charB_dir):
        hashvalue=hashlib.md5(filename.encode('utf-8')).hexdigest()
        if hashvalue in md5dict:
            shutil.copytree(os.path.join(charA_dir,filename),os.path.join(dataA_dir,filename))
            shutil.copytree(os.path.join(charB_dir,filename),os.path.join(dataB_dir,filename))


def UsefulDataset(srcDir,tarDir,charname):

    charDir1=os.path.join(srcDir, charname[0])
    charDir2=os.path.join(srcDir, charname[1])
    motions=os.listdir(charDir1)

    for motion in motions:
        motionDir1=os.path.join(charDir1, motion)
        motionDir2=os.path.join(charDir2, motion)
        pathDir1 = os.listdir(motionDir1)

        filenumber=len(pathDir1)
        rate=0.3
        picknumber=int(filenumber*rate)
        sample = random.sample(pathDir1, picknumber)

        for name in sample:
            tarPath1=os.path.join(tarDir,charname[0],motion)
            if not os.path.exists(tarPath1):
                os.makedirs(tarPath1)
            shutil.copy(os.path.join(motionDir1,name), os.path.join(tarPath1,name))

            tarPath2 = os.path.join(tarDir, charname[1], motion)
            if not os.path.exists(tarPath2):
                os.makedirs(tarPath2)
            shutil.copy(os.path.join(motionDir2, name), os.path.join(tarPath2, name))


def selectDataset(fileDir1,fileDir2,tarDir1,tarDir2):

    if not os.path.exists(tarDir1):
        os.makedirs(tarDir1)
    if not os.path.exists(tarDir2):
        os.makedirs(tarDir2)

    pathDir = os.listdir(fileDir1)
    filenumber = len(pathDir)
    rate = 0.5
    picknumber = int(filenumber * rate)

    sample = random.sample(pathDir, picknumber)
    other=list(set(pathDir).difference(set(sample)))

    for name in sample:
        shutil.move(os.path.join(fileDir1,name), os.path.join(tarDir1,name))

    for name in other:
        shutil.move(os.path.join(fileDir2,name), os.path.join(tarDir2,name))


def movefiles(Dir,character_names):
    for char in character_names:
        i=0
        char_dir=os.path.join(Dir, char)
        animation_names = os.listdir(char_dir)
        animation_names.sort()

        for anim in animation_names:
            anim_path = os.path.join(char_dir, anim)
            file_names = os.listdir(anim_path)
            file_names.sort()

            for file in file_names:
                path=os.path.join(anim_path, file)          #The path of json file

                with open(path,'r') as pose:
                    json_f=json.load(pose)

                save_dir = os.path.join(Dir,char)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, '%04d.json' % (i+1))
                with open(save_path, 'w') as f:
                    json.dump(json_f, f)
                i = i + 1
            shutil.rmtree(anim_path)


def poseNorm(Dir,character_names):

    njoints = []  # all joints position
    njoints_bias = []
    allnum_char = []


    for char in character_names:
        char_dir=os.path.join(Dir, char)
        file_names = os.listdir(char_dir)
        file_names.sort()
        num_char = file_names.__len__()
        allnum_char.append(num_char)

        for file in file_names:
            path=os.path.join(char_dir, file)          #The path of json file

            with open(path,'r') as pose:
                joints_position = np.array(json.load(pose)['pose_keypoints_2d']).reshape((15, 2))
                njoints.append(joints_position)
                joints_bias = joints_position - bbox_mid(joints_position)
                njoints_bias.append(joints_bias)

    njoints = np.array(njoints)
    njoints_bias = np.array(njoints_bias)
    maxLength = getMaxLength(njoints)
    ntgt_jointpos = (H * njoints_bias) / maxLength + H / 2
    num_pose = ntgt_jointpos.shape[0]

    for i in range(num_pose):
        tgt_jointpos = ntgt_jointpos[i, :, :]
        jbias = njoints_bias[i,:,:]
        pose_out = {"pose_keypoints_2d": jbias.flatten().tolist()}
        json_out = {"pose_keypoints_2d": tgt_jointpos.flatten().tolist()}
        if i in range(allnum_char[0]):
            save_dir=os.path.join(Dir, 'norm', character_names[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path_p = os.path.join(Dir, character_names[0], '%04d.json' % (i+1))
            save_path_j = os.path.join(save_dir, '%04d.json' % (i+1))
        else:
            save_dir=os.path.join(Dir, 'norm', character_names[1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path_p = os.path.join(Dir, character_names[1], '%04d.json' % (i-allnum_char[0]+1))
            save_path_j = os.path.join(save_dir, '%04d.json' % (i-allnum_char[0]+1))

        with open(save_path_j, 'w') as f:
            json.dump(json_out, f)
        with open(save_path_p, 'w') as f:
            json.dump(pose_out, f)

    with open(maxlength_path, 'w') as f:
        json.dump(maxLength, f)


def divideData(dataDir,gtDir,trainDir,testDir,gtTrain,gtTest,charname1,charname2):

    charDir=os.path.join(dataDir, charname1)
    charDirGt=os.path.join(gtDir, charname2)
    pathDir = os.listdir(charDir)
    filenumber=len(pathDir)
    rate=0.8
    picknumber=int(filenumber*rate)

    trainPath = os.path.join(trainDir, charname1)
    gtTrainPath = os.path.join(gtTrain, charname2)
    testPath = os.path.join(testDir, charname1)
    gtTestPath = os.path.join(gtTest, charname2)

    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(gtTrainPath):
        os.makedirs(gtTrainPath)

    if not os.path.exists(testPath):
        os.makedirs(testPath)
    if not os.path.exists(gtTestPath):
        os.makedirs(gtTestPath)

    for i in range(1,picknumber+1):     # train
        shutil.copy(os.path.join(charDir,'%04d.json' % i), os.path.join(trainPath,'%04d.json' % i))
        shutil.copy(os.path.join(charDirGt,'%04d.json' % i), os.path.join(gtTrainPath,'%04d.json' % i))

    for i in range(picknumber+1,filenumber+1):     # test
        shutil.copy(os.path.join(charDir,'%04d.json' % i), os.path.join(testPath,'%04d.json' % i))
        shutil.copy(os.path.join(charDirGt,'%04d.json' % i), os.path.join(gtTestPath,'%04d.json' % i))


def main():

    print("Start Create Data...")
    character_names = os.listdir(ALL_JSON3D_DIR)

    selectSameMotion(ALL_JSON3D_DIR,SAMEDATA3D_DIR,character_names) # select same motion of two characters from All Dataset as Dataset
    print("selectSameMotion is Done.")

    UsefulDataset(SAMEDATA3D_DIR,USEDATA3D_DIR,character_names)     # select useful data(different pose) as Dataset
    print("UsefulDataset is Done.")

    shutil.copytree(USEDATA3D_DIR,DATA3D_DIR)

    DataA_dir=os.path.join(DATA3D_DIR, character_names[0])
    DataB_dir=os.path.join(DATA3D_DIR, character_names[1])
    GtA=os.path.join(GT3D_DIR, character_names[0])
    GtB=os.path.join(GT3D_DIR, character_names[1])

    # select different pose as dataset of character A and B
    selectDataset(DataA_dir,DataB_dir,GtA,GtB)
    print("selectDataset is Done.")

    movefiles(DATA3D_DIR,character_names)
    movefiles(GT3D_DIR,character_names)
    print("movefiles is Done.")

    joint3Dto2D.main(DATA3D_DIR,DATA2D_DIR,character_names)
    joint3Dto2D.main(GT3D_DIR,GT2D_DIR,character_names)
    print("pose projection is Done.")

    poseNorm(DATA2D_DIR, character_names)
    poseNorm(GT2D_DIR, character_names)
    print("pose normalization is Done.")

    # divide 2D dataset into train set,test set,(validation set)
    divideData(DATA2D_DIR, GT2D_DIR, TRAIN2D_DIR, TEST2D_DIR, GT_TRAIN2D_DIR, GT_TEST2D_DIR, character_names[0],
               character_names[1])
    divideData(DATA2D_DIR, GT2D_DIR, TRAIN2D_DIR, TEST2D_DIR, GT_TRAIN2D_DIR, GT_TEST2D_DIR, character_names[1],
               character_names[0])
    print("divide 2D dataset is Done.")

    # divide 3D dataset into train set,test set,(validation set)
    divideData(DATA3D_DIR, GT3D_DIR, TRAIN3D_DIR, TEST3D_DIR, GT_TRAIN3D_DIR, GT_TEST3D_DIR, character_names[0],
               character_names[1])
    divideData(DATA3D_DIR, GT3D_DIR, TRAIN3D_DIR, TEST3D_DIR, GT_TRAIN3D_DIR, GT_TEST3D_DIR, character_names[1],
               character_names[0])
    print("divide 3D dataset is Done.")

    print("Create Done.")


if __name__ == '__main__':
    main()