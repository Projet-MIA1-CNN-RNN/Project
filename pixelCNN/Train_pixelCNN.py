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
from torch.autograd import Variable
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



""""
for batch_idx, (X,Y) in enumerate(data_test):
	  print(batch_idx)

for images, labels in data_test:
	print(labels)
"""


# Hyperparameters
model_MNIST = PixelCNN()

# Hyperparameters for MNIST 
batch_size = 2**10
epochs = 4
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model_MNIST.parameters(),lr=0.001,alpha=0.9)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model = model.to(device)
    for batch, (X, y) in enumerate(dataloader):
        target = Variable(X[:,0,:,:]*255)
        X = X.to(device)
        target = target.to(device)
        # Compute prediction and loss
        pred = model(X*255)
        pred = pred[:,0,:,:] # DOUTE ICI A CORRIGER OU ECLAIRCIR
        #print(pred)
        loss = loss_fn(pred,target)

        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(data_train,model_MNIST,loss_fn,optimizer)
print("Done!")

#Saving the network
PATH = './mnist_local.pth'
torch.save(model_MNIST.state_dict(), PATH)