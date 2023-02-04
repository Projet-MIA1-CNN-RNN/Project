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




class PixelCNN(nn.Module):
	def __init__(self, nb_layer_block=12, channels=64, device=None):
		super(PixelCNN, self).__init__()
		self.nb_block = nb_layer_block
		self.channels = channels
		self.layers = {}
		self.device = device
		#first convolution (cf TABLE 1)
		self.ConvA = nn.Sequential(
			MaskedConv2D('A',1, 2*self.channels, kernel_size=7,padding="same", bias=False),
			nn.ReLU(True)
		)
		#Residual blocks for PixelCNN (figure 5)
		self.multiple_blocks = nn.Sequential(
			nn.Conv2d(2*self.channels, self.channels, kernel_size = 1,padding='same', bias=False),
			nn.ReLU(True),
			MaskedConv2D('B',self.channels,self.channels, kernel_size = 3,padding='same', bias=False),
			nn.ReLU(True),
			nn.Conv2d(self.channels,2*self.channels, kernel_size = 1,padding='same', bias=False),
			nn.ReLU(True)
		)
		#finalisation
		self.end = nn.Sequential(
			nn.ReLU(True),
			MaskedConv2D('B',2*self.channels, 2 * self.channels, padding ='same',kernel_size = 1, bias=False),
			nn.ReLU(True),
			MaskedConv2D('B',2* self.channels,2 * self.channels,padding='same', kernel_size = 1, bias=False),
			nn.Sigmoid()
		)

	def residual_block(self, x):
		return (x + self.multiple_blocks(x))
	
	def forward(self,x,**kwargs):
		x = self.ConvA(x)
		for i in range(self.nb_block):
			x = self.residual_block(x)
		x = self.end(x)
		return x

model = PixelCNN()