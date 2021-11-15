import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,6,5,padding=2),
                                 nn.MaxPool2d(2,2),
                                )
        self.conv2=nn.Sequential(nn.Conv2d(6,16,5),
                                 nn.MaxPool2d(2,2),
                                 )
        self.dense=nn.Sequential(nn.Linear(16*5*5,120),
                                 nn.Linear(120,84),
                                 nn.Linear(84,10))

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=torch.flatten(x,1)
        x=self.dense(x)
        return x
net=LeNet()
print(net)