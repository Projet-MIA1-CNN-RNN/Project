
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from pixelCNN import PixelCNN


def test_MNIST():

    PATH = 'pixelCNN\mnist.pth'
    assert os.path.exists(PATH), 'Saved Model File Does not exist!'
    no_images = 36
    images_size =  28
    images_channels =  1
    #Define and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PixelCNN().to(device)
    if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')),)
    model.eval()
    
    sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
    sample.fill_(0)

    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            out = model(sample)

            sample[:,:,i,j] = torch.multinomial(probs, 1).float()
            #print(sample[:,:,i,j])
    print(out)
    #Saving images row wise
    torchvision.utils.save_image(sample, 'pixelCNN\im_test.png', nrow=6, padding=0)

    pass


test_MNIST()
