#custom module to create dataset

import torch
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from components.transform_albumentation import data_albumentation

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, train=True, batch_size=64):
        train_transform, test_transform = data_albumentation(horizontalflip_prob=0.5,
                                                                 rotate_limit=15,
                                                                 shiftscalerotate_prob=0.25,
                                                                 num_holes=1,cutout_prob=0.5)
        self.cuda = torch.cuda.is_available()
        self.required_transform = train_transform if train else test_transform
        self.dataset = getattr(datasets, dataset_name)(root='./data',
                                                       download=True, train=train, transform=self.required_transform)

        self.dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)
    
        self.dataloader = torch.utils.data.DataLoader(self.dataset, **self.dataloader_args)  #returns dataloader when .dataloader is called on Dataset instance.


    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)
