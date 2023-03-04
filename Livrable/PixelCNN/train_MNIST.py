import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
import PixelCNN_MNIST
from PixelCNN_MNIST import PixelCNN_MNIST,device

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x>0,
                       lambda x: x.float(),
            ])),
    batch_size=batch_size, shuffle=True,pin_memory=True)

x, _ = train_loader.dataset[7777]
plt.imshow(x.numpy()[0], cmap='gray')



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model = model.to(device)
    for batch, (X, y) in enumerate(dataloader):
        target = Variable(X[:,:,:,:])
        X = X.to(device)
        target = target.to(device)
        # Compute prediction and loss
        pred = model(X)
        pred = pred[:,:,:,:] 
        #print(pred)
        
        loss = loss_fn(pred,target)

        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()

        if batch % 1500 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


model_MNIST = PixelCNN_MNIST()

# Hyperparameters for MNIST 
epochs = 4
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model_MNIST.parameters(),lr=0.001,alpha=0.9)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader,model_MNIST,loss_fn,optimizer)
print("Done!")


PATH = './cifar_kaggle.pth'
torch.save(model_cifar.state_dict(), PATH)