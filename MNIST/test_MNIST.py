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
import os
from PIL import Image


def test_MNIST(model, weight_path ,no_images ,n_col ,n_row, user=True ):

    #PATH = '/kaggle/working/mnist_kaggle.pth'
    #assert os.path.exists(weight_path), 'Saved Model File Does not exist!'
    images_size =  28
    images_channels =  1

    #Define and load model
    model = PixelCNN_MNIST().to(device)
    global init_param
    init_param = model.parameters()
    model.load_state_dict(torch.load(weight_path,map_location=device))
    model.eval()
    test_param = model.parameters()
    sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
    sample.fill_(0)

    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            out = model(sample)
            probs = torch.sigmoid(out[:,:,i,j]).data
            results = torch.bernoulli(probs)
            sample[:,:,i,j] = results
    #Saving images row wise
    if user:
        torchvision.utils.save_image(sample,'pages/generated_images/user_model_image.png',
                                     ncol=n_col, nrow=n_row, padding=0)
    else:
        torchvision.utils.save_image(sample,'pages/generated_images/default_model_image.png',
                                     ncol=n_col, nrow=n_row, padding=0)
    pass
