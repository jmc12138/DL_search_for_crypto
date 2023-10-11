from .models import register
import torch

import numpy as np
import random


import torch.nn as nn

# Deterministic random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


@register('MLP_speck1')
class Net(nn.Module):
    def __init__(self,dropout):

        super(Net, self).__init__()


        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(640,1280)
        self.fc2 = nn.Linear(1280, 1280*2)
        self.fc3 = nn.Linear(1280*2, 1280)
        self.fc4 = nn.Linear(1280, 640)
        self.fc5 = nn.Linear(640, 64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):

        x = x.flatten(1,2)
        x = self.fc1(x)
        x = self.leaky_relu(x)        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = self.fc5(x)
        x = self.leaky_relu(x)
        x = self.fc6(x)

        # x = self.dropout(x)
        # x = self.norm2(x)
        return x
