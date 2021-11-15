import torch
from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        features=nn.Sequential()
        features.add_module('conv1',nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4))
        features.add_module('pool1',nn.MaxPool2d(kernel_size=3,stride=2))
        features.add_module('relu',nn.ReLU())
        features.add_module('conv2',nn.Conv2d(96,256,5,padding=2))
        features.add_module('pool2',nn.MaxPool2d(kernel_size=3,stride=2))
        features.add_module('relu',nn.ReLU())
        features.add_module('conv3',nn.Conv2d(256,384,3,padding=1))
        features.add_module('relu',nn.ReLU())
        features.add_module('conv4',nn.Conv2d(384,384,3,padding=1))
        features.add_module('relu',nn.ReLU())
        features.add_module('conv5',nn.Conv2d(384,256,3,padding=1))
        features.add_module('pool3',nn.MaxPool2d(kernel_size=3,stride=2))
        features.add_module('relu',nn.ReLU())
        