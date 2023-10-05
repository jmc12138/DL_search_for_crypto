import torch.nn as nn
import torch
from .models import register
import torch.nn.functional as F


@register('VulCNN')
class TextCNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)           
        self.num_filters = 32                                        
        # classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        # self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(hidden_state)
        out = self.fc(out)
        # return out, hidden_state
        return out