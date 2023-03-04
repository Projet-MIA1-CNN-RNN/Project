import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from MaskedConv import MaskedConv2D,device


class PixelCNN_MNIST(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, nb_layer_block=12, 
                 h_channels=32, device=None):
        super(PixelCNN_MNIST, self).__init__()
        
        torch.cuda.empty_cache()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_block = nb_layer_block
        self.h_channels = h_channels
        self.layers = {}
        self.device = device
        #first convolution (cf TABLE 1)
        self.ConvA = nn.Sequential(
            MaskedConv2D('A',in_channels, 2*self.h_channels, kernel_size=7,padding="same", bias=True),
            nn.ReLU(True)
        )
        #Residual blocks for PixelCNN (figure 5)
        self.multiple_blocks = nn.Sequential(
            nn.Conv2d(2*self.h_channels, self.h_channels, kernel_size = 1,padding='same', bias=True),
            nn.ReLU(True),
            MaskedConv2D('B',self.h_channels,self.h_channels, kernel_size = 3,padding='same', bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.h_channels,2*self.h_channels, kernel_size = 1,padding='same', bias=True),
            nn.ReLU(True)
        )
        #finalisation
        
        self.end_mnist = nn.Sequential(
            MaskedConv2D('B',2*self.h_channels, 2 * self.h_channels, padding ='same',kernel_size = 1, bias=True),
            nn.ReLU(True),
            MaskedConv2D('B',2* self.h_channels,2*self.h_channels,padding='same', kernel_size = 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(2*self.h_channels,self.out_channels,padding='same',kernel_size = 1, bias=True),
        )

    def residual_block(self, x):
        return (x + self.multiple_blocks(x))

    def forward(self,x,**kwargs):
        x = self.ConvA(x)

        for i in range(self.nb_block):
            x = self.residual_block(x)
        
        x = self.end_mnist(x)


        return x