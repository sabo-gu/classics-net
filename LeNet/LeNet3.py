import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        layer1=nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(1,6,5,padding=2))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        layer1.add_module('relu',nn.ReLU())
        layer1.add_module('conv2',nn.Conv2d(6,16,5))
        layer1.add_module('pool2',nn.MaxPool2d(2,2))
        layer1.add_module('relu',nn.ReLU())
        layer1.add_module('flatten',nn.Flatten(1))
        layer1.add_module('dense1',nn.Linear(16*5*5,120))
        layer1.add_module('dense2',nn.Linear(120,84))
        layer1.add_module('dense3',nn.Linear(84,10))
        self.layer1=layer1

    def forward(self,x):
        x=self.layer1(x)
        return x
net=LeNet()
print(net)
import torchvision
torchvision.models.AlexNet