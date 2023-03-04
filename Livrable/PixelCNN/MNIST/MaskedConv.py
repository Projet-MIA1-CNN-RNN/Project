import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


class MaskedConv2D(nn.Conv2d):
    def __init__(self,mask_type, *args, **kwargs):
        super(MaskedConv2D, self).__init__(*args, **kwargs)
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        self.register_buffer('mask', self.weight.data.clone())
        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0

    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedConv2D, self).forward(x)