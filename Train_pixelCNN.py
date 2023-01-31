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

data_test = torch.utils.data.DataLoader(
	MNIST(
		  '~/mnist_data', train=False, download=True, 
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
model = PixelCNN()

# Hyperparameters
learning_rate = 0.01
batch_size = 2**10
epochs = 2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters())

def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		target = Variable(X[:,0,:,:]*255)
		print(target.size())
		X = X.to(device)
		target = target.to(device)
		# Compute prediction and loss
		pred = model(X)
		pred = pred[:,-1,:,:]
		loss = loss_fn(pred,target)

		# Backpropagation (always in three steps)
		optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	


'''
loss_overall = []
for i in range(epochs):

		
		model.train(True)
		step = 0
		loss_= 0

		for images, labels in data_train:
			
			target = Variable(images[:,0,:,:]*255).long()
			images = images.to(device)
			target = target.to(device)
			
			


			optimizer.zero_grad()

			output = model(images)
			loss = loss_fn(output)
			loss.mean().backward()
			optimizer.step()


			loss_+=loss
			step+=1

			if(step%100 == 0):
				print('Epoch:'+str(i)+'\t'+ str(step) +'\t Iterations Complete \t'+'loss: ', loss.item()/1000.0)
				loss_overall.append(loss_/1000.0)
				loss_=0
		print('Epoch: '+str(i)+' Over!')

'''

for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(data_train,model,loss_fn,optimizer)
print("Done!")

#Saving the network
PATH = './mnist.pth'
torch.save(model.state_dict(), PATH)