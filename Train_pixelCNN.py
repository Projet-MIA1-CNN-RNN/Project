import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from pixelCNN import PixelCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

data_train = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=True, download=True, 
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=64,
          shuffle=True
          )

data_test = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=False, download=True, 
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=64,
          shuffle=True
          )

