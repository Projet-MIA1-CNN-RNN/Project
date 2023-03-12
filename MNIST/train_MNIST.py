import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PixelCNN_MNIST_model import PixelCNN_MNIST,device
from torchvision.datasets import MNIST
from torch.autograd import Variable


batch_size = 16
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    lambda x: x>0,
                    lambda x: x.float(),
            ])),
    batch_size=batch_size, shuffle=True,pin_memory=True)



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model = model.to(device)
    list_loss = []
    print("test")
    for batch, (X, y) in enumerate(dataloader):
        target = Variable(X[:,:,:,:])
        X = X.to(device)
        target = target.to(device)
        # Compute prediction and loss
        pred = model(X)
        pred = pred[:,:,:,:] 
        #print(pred)
        
        loss = loss_fn(pred,target)
        list_loss.append(loss)
        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()
    return np.mean(list_loss)


model_MNIST = PixelCNN_MNIST()

# Hyperparameters for MNIST 
epochs = 4
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.001
alpha = 0.9
batch_size = 16
optimizer = torch.optim.RMSprop(model_MNIST.parameters(),lr=lr,alpha=alpha)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    iter = train_loop(train_loader,model_MNIST,loss_fn,optimizer)
print("Done!")

list_epoch = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    accuracy = train_loop(train_loader,model_MNIST,loss_fn,optimizer)
    list_epoch.append(accuracy)
    print(accuracy)
print("Done!")

plt.plot(epochs,accuracy)

PATH = './cifar_kaggle.pth'
torch.save(model_MNIST.state_dict(), PATH)