## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) #224*224*32
        self.conv2 = nn.Conv2d(32, 64, 5) #110*110*64
        self.conv3 = nn.Conv2d(64, 128, 5) #53*53*128
        self.conv4 = nn.Conv2d(128,256,5) #24*24*256
        
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(10*10*256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
        self.dropout = nn.Dropout(0.35)
        
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        
        x = self.maxpool(self.conv1(x))
        x = self.bn2(self.maxpool(self.conv2(x)))
        x = self.bn3(self.maxpool(self.conv3(x)))
        x = self.bn4(self.maxpool(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
