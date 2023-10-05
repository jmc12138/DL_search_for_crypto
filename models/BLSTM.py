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


@register('BLSTM_DES1')
class Net(nn.Module):
    def __init__(self,dropout,hidden_size,num_layers):
        self.hidden_size = hidden_size
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=self.hidden_size, num_layers=num_layers,bidirectional=True,batch_first=True)


        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        self.norm1 = nn.LayerNorm(self.hidden_size * 2)
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:,-1,:].contiguous().view(-1, self.hidden_size * 2)
        # x = self.norm1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        # x = self.dropout(x)

        # x = self.norm2(x)

        x = self.fc2(x)

        return x


@register('BLSTM_DES2')
class Net(nn.Module):
    def __init__(self,dropout,hidden_size,num_layers):
        self.hidden_size = hidden_size
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=self.hidden_size, num_layers=num_layers,bidirectional=True,batch_first=True)


        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        self.norm1 = nn.LayerNorm(self.hidden_size * 2)
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:,-1,:].contiguous().view(-1, self.hidden_size * 2)
        # x = self.norm1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        # x = self.dropout(x)

        # x = self.norm2(x)

        x = self.fc2(x)

        return x



@register('BLSTM_DES4')
class Net(nn.Module):
    def __init__(self,dropout,hidden_size,num_layers):
        self.hidden_size = hidden_size
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=self.hidden_size, num_layers=num_layers,bidirectional=True,batch_first=True)


        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        self.norm1 = nn.LayerNorm(self.hidden_size * 2)
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:,-1,:].contiguous().view(-1, self.hidden_size * 2)
        # x = self.norm1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        # x = self.dropout(x)

        # x = self.norm2(x)

        x = self.fc2(x)

        return x
