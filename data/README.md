# EVA-8_Phase-1_Assignment-6
This is the assignment of 6th session in phase-1 of EVA-8 from TSAI.

## Introduction

### Objective
Objective of this assignment is to build three CNN based network which will take [CIFAR10 Dataset](<http://yann.lecun.com/exdb/mnist/>) and should have following properties.
1. This three networks should be following:
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

