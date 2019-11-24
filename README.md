## RGBD Segmentation toolbox

This is a basic framework for RGBD Segmentation task, built in pytorch.

The baseline model is dilated ResNet101 without using depth information. Easy to expand.

**Highlights**

- Synchronous BN
- Fewness of Training Time
- Better Reproduced Performance
- Processed NYUD dataset (image, label, hha, XYZ coordinates) 

 ### 1. Requirements

- pytorch == 0.4.0 please refer to https://github.com/pytorch/pytorch#installation.

- PIL 

- opencv

- scipy == 1.2.0

- python 3.5

### 2. Run the code

**Dataset and pretrained model**

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

Please download processed NYUDv2 Dataset [NYUDataset.zip]( https://pan.baidu.com/s/1mFG6Skq6lwRAiNafRfKXjQ) (code k48c), and unzip it into `dataset` folder.

**Compiling**

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py
```

The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

**Training and Evaluation**

```bash
python train.py
```

### 3. Results

![](figure\result.png)

| Model    | Acc. | mAcc. | mIoU |
| -------- | ---- | ----- | ---- |
| Baseline | 71.1 | 54.6  | 42.2 |

### Thanks to the Third Party Libs
[inplace_abn](https://github.com/mapillary/inplace_abn) - 
[Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab) - 
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
[PyTorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)
[Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python)