# Pose Retargeting (using CycleGAN)
This repository implements a CycleGAN mothod use in pose retargeting.
## Required Software
To run it, you have to install following sofware with specified version(I would hignly recommend that create a virtual environment first):  
pytorch==0.4.1   
python>=3.6 

## Getting Started
### Prepare Data
- Download Mixamo Data

I use the [Mixamo dataset](https://www.mixamo.com/#/), and I put my dataset in __Data__ folder.If you want to collect Mixamo Data by yourself, you can follow the guide [here](https://github.com/DetectiveLYJ/Pose-Retargeting-using-CycleGAN-/blob/master/Data/guide%20for%20dataset%20downloading.md).

- Preprocess the downloaded data
```
cd GycleGAN_Retargeting
cd Code
blender --backgrounde -P ./data/fbx2json3D.py
python ./data/CreateData.py
```

### train
- Train the model on GPU:
```
python train.py --dataroot ../Data/Dataset --name baseline --model cycle_gan
```

- If you want to use a identity loss:
```
python train.py --dataroot ../Data/Dataset --name baseline --model cycle_gan --lambda_idt 0.5
```

### test
- Use a pretrained model:
```
python train.py --dataroot ../Data/Dataset/testA --name baseline --direction BtoA --model test
```

- If you want to compare with other method:
```
python train.py --dataroot ../Data/Dataset/testA --name baseline --direction BtoA --model test --compare 1 
```


