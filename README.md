# EVA-8_Phase-1_Assignment-6
This is the assignment of 6th session in phase-1 of EVA-8 from TSAI.

## Introduction

### Objective
Objective of this assignment is to build a CNN based network which will take [CIFAR10 Dataset](<http://yann.lecun.com/exdb/mnist/>) and should have following properties.
1. This network architecture should have following properties:
    - Network should not use any Max Pooling, feature map size reduction should be done by convolution process.
    - Receptive Field of the model should be upto 44x44.
    - one of the layers must use Depthwise Separable Convolution
    - one of the layers must use Dilated Convolution
2. use albumentation library and apply: horizontal flip, shiftScaleRotate, coarseDropout
3. Model should achieve accuracy of upto 85% on CIFAR10 dataset.
2. We should have single model.py file which will contain model architecture defination as well as other modularized components.
3. Module should have less then 200k parameters in total.

### Repository setup
Since all the essential modules are written in .py files which are then getting called in main notebook, it is necessary to understand the structure of the repository.
Below is a quick look on how the repository is setup:
<br>
```
EVA-8_Phase-1_Assignment-6/
  |
  ├── EVA_Assignment-6.ipynb    <- Main colab notebook which will call other modules and perform training
  |
  ├── README.md                           <- The top-level README for developers using this project.
  |
  ├── LICENSE                             <- Standered apache 2.0 based license
  |
  ├── component/
  │   ├── data.py             <- Python file to download, process and create dataset for training.
  │   ├── model.py            <- Python file where model arcitecture is defined and can be changed if required.
  │   ├── training.py         <- Python file where training code is defined. Forward and backward pass will be done by this.
  │   ├── test.py             <- File to perform evaluation while training the model. It on performs forward pass with no gradient calc.
  │   ├── albumetation.py     <- We define albumentation based augmentation here.
  │   └── plot_util.py        <- Contains utility function to plot graphs and images.
  │
  ├── data/                   <- Directory to store data downloaded via torchvision
  │   ├── MNIST               <- Example when MNIST data will be downloaded
  │   ├── CIFAR10             <- Example when ImageNet data will be downloaded.
  │
  ├── reports/                <- Directory to store reports/results/etc.
  │   └── figures             <- Generated graphics and figures to be used in reporting
  |
  ├── repo_util/              <- Folder containing all required artifacts for the README.md
```
### Getting started
To get started with the repo is really easy. Follow below steps to run everything by your self:
1. Open main notebook `EVA_Assignment-6.ipynb` and click on "open with cloab" option.
2. Run the first cell. This cell will git clone this repo so that all fucntions are available in your runtime.
3. That's it! Now you can execute the cell and train this three models. Other detials are commented in the main notebook.

## Data representation
In this assignment I am using [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) with this dataset I am applying following augmentation on the top:
1. `HorizontalFlip` - Fliping the image along horizontal axis.
2. `ShiftscaleRotate` - Perform transformation randomly, this transformation includes shifting of image, scaling and rotating.
3. `CoarseDropOut` - Overlay a rectangle patch(half the size of original image) on a image randomly. (simulate object hindarence)
4. `ColorJitter` - Randomly changes the brightness, contrast, and saturation of an image. (simulate lighting condition)
5. `ToGray` - Randomly change RBG to gray-scale. 
6. `Normalize` - Normalize image i.e. zero centring (zero mean) and scaling (one std)

Below is the graph representing the input training dataset after appling all augmentations c .
![Alt text](repo_util/data_6.JPG?raw=true "model architecture")

## Model representation
In this assignment, I am using two different architecture `Netv1()` and `Netv2()`. Below are more discription on each model used in the training.
- `Netv1()` - This architecture mainly uses 3x3-kernel convolutaion of stride 2 to reduce the shape of feature map instead of using Max-Pooling. Netv1() model architecture is define inside `components/model.py` inside `Netv1()` class. This architecture also all the requirements as mentioned.
- `Netv2()` - This architecture mainly uses uses 5x5-kernel with dilation of 2 and stride of 1 to reduce the shape of the feature map instead of Max-pooling.

Below is a snapshot of model arcitecture for both model `netV1()` and `netV2()`
```
-----------------------------------Netv1() arcitecture with 3x3 stride 2 pooling------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 30, 30]           4,608
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
           Dropout-8           [-1, 32, 30, 30]               0
            Conv2d-9           [-1, 32, 30, 30]             288
           Conv2d-10           [-1, 32, 30, 30]           1,024
DepthwiseSeparable-11           [-1, 32, 30, 30]               0
      BatchNorm2d-12           [-1, 32, 30, 30]              64
             ReLU-13           [-1, 32, 30, 30]               0
          Dropout-14           [-1, 32, 30, 30]               0
           Conv2d-15           [-1, 32, 15, 15]           9,216
      BatchNorm2d-16           [-1, 32, 15, 15]              64
             ReLU-17           [-1, 32, 15, 15]               0
          Dropout-18           [-1, 32, 15, 15]               0
           Conv2d-19           [-1, 32, 15, 15]           9,216
      BatchNorm2d-20           [-1, 32, 15, 15]              64
             ReLU-21           [-1, 32, 15, 15]               0
          Dropout-22           [-1, 32, 15, 15]               0
           Conv2d-23           [-1, 64, 13, 13]          18,432
      BatchNorm2d-24           [-1, 64, 13, 13]             128
             ReLU-25           [-1, 64, 13, 13]               0
          Dropout-26           [-1, 64, 13, 13]               0
           Conv2d-27           [-1, 64, 13, 13]             576
           Conv2d-28           [-1, 64, 13, 13]           4,096
DepthwiseSeparable-29           [-1, 64, 13, 13]               0
      BatchNorm2d-30           [-1, 64, 13, 13]             128
             ReLU-31           [-1, 64, 13, 13]               0
          Dropout-32           [-1, 64, 13, 13]               0
           Conv2d-33             [-1, 64, 7, 7]          36,864
      BatchNorm2d-34             [-1, 64, 7, 7]             128
             ReLU-35             [-1, 64, 7, 7]               0
          Dropout-36             [-1, 64, 7, 7]               0
           Conv2d-37             [-1, 64, 7, 7]          36,864
      BatchNorm2d-38             [-1, 64, 7, 7]             128
             ReLU-39             [-1, 64, 7, 7]               0
          Dropout-40             [-1, 64, 7, 7]               0
           Conv2d-41             [-1, 64, 5, 5]          36,864
      BatchNorm2d-42             [-1, 64, 5, 5]             128
             ReLU-43             [-1, 64, 5, 5]               0
          Dropout-44             [-1, 64, 5, 5]               0
           Conv2d-45             [-1, 64, 5, 5]             576
           Conv2d-46             [-1, 64, 5, 5]           4,096
DepthwiseSeparable-47             [-1, 64, 5, 5]               0
      BatchNorm2d-48             [-1, 64, 5, 5]             128
             ReLU-49             [-1, 64, 5, 5]               0
          Dropout-50             [-1, 64, 5, 5]               0
           Conv2d-51             [-1, 32, 7, 7]           2,048
      BatchNorm2d-52             [-1, 32, 7, 7]              64
             ReLU-53             [-1, 32, 7, 7]               0
          Dropout-54             [-1, 32, 7, 7]               0
           Conv2d-55             [-1, 32, 5, 5]           9,216
      BatchNorm2d-56             [-1, 32, 5, 5]              64
             ReLU-57             [-1, 32, 5, 5]               0
          Dropout-58             [-1, 32, 5, 5]               0
           Conv2d-59             [-1, 32, 5, 5]             288
           Conv2d-60             [-1, 32, 5, 5]           1,024
DepthwiseSeparable-61             [-1, 32, 5, 5]               0
      BatchNorm2d-62             [-1, 32, 5, 5]              64
             ReLU-63             [-1, 32, 5, 5]               0
          Dropout-64             [-1, 32, 5, 5]               0
           Conv2d-65             [-1, 10, 5, 5]             320
        AvgPool2d-66             [-1, 10, 1, 1]               0
================================================================
Total params: 177,296
Trainable params: 177,296
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.39
Params size (MB): 0.68
Estimated Total Size (MB): 5.07
----------------------------------------------------------------
-----------------------------------Netv2() arcitecture with dilated conv pooling------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 30, 30]           4,608
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
           Dropout-8           [-1, 32, 30, 30]               0
            Conv2d-9           [-1, 32, 30, 30]             288
           Conv2d-10           [-1, 32, 30, 30]           1,024
DepthwiseSeparable-11           [-1, 32, 30, 30]               0
      BatchNorm2d-12           [-1, 32, 30, 30]              64
             ReLU-13           [-1, 32, 30, 30]               0
          Dropout-14           [-1, 32, 30, 30]               0
           Conv2d-15           [-1, 32, 30, 30]           1,024
      BatchNorm2d-16           [-1, 32, 30, 30]              64
             ReLU-17           [-1, 32, 30, 30]               0
          Dropout-18           [-1, 32, 30, 30]               0
           Conv2d-19           [-1, 32, 24, 24]          25,600
      BatchNorm2d-20           [-1, 32, 24, 24]              64
             ReLU-21           [-1, 32, 24, 24]               0
          Dropout-22           [-1, 32, 24, 24]               0
           Conv2d-23           [-1, 32, 24, 24]           9,216
      BatchNorm2d-24           [-1, 32, 24, 24]              64
             ReLU-25           [-1, 32, 24, 24]               0
          Dropout-26           [-1, 32, 24, 24]               0
           Conv2d-27           [-1, 64, 22, 22]          18,432
      BatchNorm2d-28           [-1, 64, 22, 22]             128
             ReLU-29           [-1, 64, 22, 22]               0
          Dropout-30           [-1, 64, 22, 22]               0
           Conv2d-31           [-1, 64, 22, 22]             576
           Conv2d-32           [-1, 64, 22, 22]           4,096
DepthwiseSeparable-33           [-1, 64, 22, 22]               0
      BatchNorm2d-34           [-1, 64, 22, 22]             128
             ReLU-35           [-1, 64, 22, 22]               0
          Dropout-36           [-1, 64, 22, 22]               0
           Conv2d-37           [-1, 32, 22, 22]           2,048
      BatchNorm2d-38           [-1, 32, 22, 22]              64
             ReLU-39           [-1, 32, 22, 22]               0
          Dropout-40           [-1, 32, 22, 22]               0
           Conv2d-41           [-1, 32, 14, 14]          25,600
      BatchNorm2d-42           [-1, 32, 14, 14]              64
             ReLU-43           [-1, 32, 14, 14]               0
          Dropout-44           [-1, 32, 14, 14]               0
           Conv2d-45           [-1, 64, 14, 14]          18,432
      BatchNorm2d-46           [-1, 64, 14, 14]             128
             ReLU-47           [-1, 64, 14, 14]               0
          Dropout-48           [-1, 64, 14, 14]               0
           Conv2d-49           [-1, 64, 12, 12]          36,864
      BatchNorm2d-50           [-1, 64, 12, 12]             128
             ReLU-51           [-1, 64, 12, 12]               0
          Dropout-52           [-1, 64, 12, 12]               0
           Conv2d-53           [-1, 64, 12, 12]             576
           Conv2d-54           [-1, 64, 12, 12]           4,096
DepthwiseSeparable-55           [-1, 64, 12, 12]               0
      BatchNorm2d-56           [-1, 64, 12, 12]             128
             ReLU-57           [-1, 64, 12, 12]               0
          Dropout-58           [-1, 64, 12, 12]               0
           Conv2d-59           [-1, 32, 12, 12]           2,048
      BatchNorm2d-60           [-1, 32, 12, 12]              64
             ReLU-61           [-1, 32, 12, 12]               0
          Dropout-62           [-1, 32, 12, 12]               0
           Conv2d-63             [-1, 32, 6, 6]          25,600
      BatchNorm2d-64             [-1, 32, 6, 6]              64
             ReLU-65             [-1, 32, 6, 6]               0
          Dropout-66             [-1, 32, 6, 6]               0
           Conv2d-67             [-1, 32, 4, 4]           9,216
      BatchNorm2d-68             [-1, 32, 4, 4]              64
             ReLU-69             [-1, 32, 4, 4]               0
          Dropout-70             [-1, 32, 4, 4]               0
           Conv2d-71             [-1, 32, 4, 4]             288
           Conv2d-72             [-1, 32, 4, 4]           1,024
DepthwiseSeparable-73             [-1, 32, 4, 4]               0
      BatchNorm2d-74             [-1, 32, 4, 4]              64
             ReLU-75             [-1, 32, 4, 4]               0
          Dropout-76             [-1, 32, 4, 4]               0
           Conv2d-77             [-1, 10, 4, 4]             320
        AvgPool2d-78             [-1, 10, 1, 1]               0
================================================================
Total params: 192,784
Trainable params: 192,784
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.03
Params size (MB): 0.74
Estimated Total Size (MB): 9.78
----------------------------------------------------------------
```

## Training logs
Below is the snippet of last  5 epochs training logs as the model is trained for 80 epochs. Here each epoch consist of two Test set losses. One is for Netv1() and other is for Netv2().
```
EPOCH: 75
`Netv1 loss`
Loss=0.5562683939933777 Batch_id=390 Accuracy=81.61: 100%|██████████| 391/391 [00:20<00:00, 19.34it/s]
Test set: Average loss: 0.4296, Accuracy: 8536/10000 (85.36%)

`Netv2 loss`
Loss=0.6562783718109131 Batch_id=390 Accuracy=77.22: 100%|██████████| 391/391 [00:21<00:00, 17.84it/s]
Test set: Average loss: 0.4966, Accuracy: 8304/10000 (83.04%)

EPOCH: 76
Loss=0.4930347800254822 Batch_id=390 Accuracy=81.56: 100%|██████████| 391/391 [00:19<00:00, 19.92it/s]
Test set: Average loss: 0.4292, Accuracy: 8523/10000 (85.23%)

Loss=0.6153714656829834 Batch_id=390 Accuracy=77.45: 100%|██████████| 391/391 [00:22<00:00, 17.61it/s]
Test set: Average loss: 0.4963, Accuracy: 8301/10000 (83.01%)

EPOCH: 77
Loss=0.4474758505821228 Batch_id=390 Accuracy=81.32: 100%|██████████| 391/391 [00:20<00:00, 19.32it/s]
Test set: Average loss: 0.4288, Accuracy: 8536/10000 (85.36%)

Loss=0.6069265604019165 Batch_id=390 Accuracy=77.25: 100%|██████████| 391/391 [00:22<00:00, 17.13it/s]
Test set: Average loss: 0.4980, Accuracy: 8310/10000 (83.10%)

EPOCH: 78
Loss=0.5485225915908813 Batch_id=390 Accuracy=81.23: 100%|██████████| 391/391 [00:19<00:00, 19.87it/s]
Test set: Average loss: 0.4285, Accuracy: 8535/10000 (85.35%)

Loss=0.8768221139907837 Batch_id=390 Accuracy=77.46: 100%|██████████| 391/391 [00:22<00:00, 17.67it/s]
Test set: Average loss: 0.4979, Accuracy: 8304/10000 (83.04%)

EPOCH: 79
Loss=0.49773159623146057 Batch_id=390 Accuracy=81.32: 100%|██████████| 391/391 [00:19<00:00, 19.86it/s]
Test set: Average loss: 0.4282, Accuracy: 8543/10000 (85.43%)

Loss=0.6468020677566528 Batch_id=390 Accuracy=77.42: 100%|██████████| 391/391 [00:22<00:00, 17.36it/s]
Test set: Average loss: 0.4997, Accuracy: 8301/10000 (83.01%)
```

## Conclusion

