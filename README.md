# CycleGAN_Retargeting
This repository implements a CycleGAN mothod use in pose retargeting.
## Required Software
------------------------------------------------------------------------------------------
To run it, you have to install following sofware with specified version(I would hignly recommend that create a virtual environment first):  
pytorch==0.4.1   
python>=3.6 

## Getting Started
### Prepare Data
- Download Mixamo Data
I use the Mixamo dataset, and I put my dataset in Data folder.If you want to collect Mixamo Data by yourself, you can follow the guide [here](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/dataset/Guide%20For%20Downloading%20Mixamo%20Data.md).

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

If you want to use a identity loss:
```
python train.py --dataroot ../Data/Dataset --name baseline --model cycle_gan --lambda_idt 0.5
```

### test
- Use a pretrained model:
```
python train.py --dataroot ../Data/Dataset/testA --name baseline --direction BtoA --model test
```

If you want to compare with other method:
```
python train.py --dataroot ../Data/Dataset/testA --name baseline --direction BtoA --model test --compare 1 
```

## Final Results
![]()

