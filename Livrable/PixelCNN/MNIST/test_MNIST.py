import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.image as mpimg
from PixelCNN_MNIST_model import PixelCNN_MNIST,device

def test_MNIST():

    PATH = '/kaggle/working/mnist_kaggle.pth'
    assert os.path.exists(PATH), 'Saved Model File Does not exist!'
    no_images = 9
    images_size =  28
    images_channels =  1

    #Define and load model
    model = PixelCNN_MNIST().to(device)
    global init_param
    init_param = model.parameters()
    model.load_state_dict(torch.load(PATH,map_location=device))
    model.eval()
    global test_param
    test_param = model.parameters()
    sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
    sample.fill_(0)

    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            out = model(sample)
            probs = torch.sigmoid(out[:,:,i,j])
            results = torch.bernoulli(probs)
            sample[:,:,i,j] = results
    #Saving images row wise
    torchvision.utils.save_image(sample, '/kaggle/working/sample2.png',ncol=3, nrow=3, padding=0)

    pass

generated_images = test_MNIST()


plt.imshow(mpimg.imread('sample2.png'))