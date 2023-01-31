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


data_train = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=True, download=True, 
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=16,
          shuffle=True
          )

data_test = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=False, download=True, 
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=16,
          shuffle=True
          )

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

model = PixelCNN()

# Hyperparameters
learning_rate = 0.01
batch_size = 2**10
epochs = 3
loss_fn = torch.nn.Softmax()
optimizer = torch.optim.RMSprop(model.parameters())

