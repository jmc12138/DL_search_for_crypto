from datasets import make


import torch.utils.data as data
try:
    from .datasets import register
except:
    from datasets import register
import os,torch
import numpy as np
import pandas as pd

class speck(data.Dataset):
    def __init__(self, dataset_path):
        self.nclass = 2
        # Load data using Pandas
        self.df = pd.read_pickle(dataset_path)
        # Extract features and labels
        self.values,self.labels = self._data_labels()
    def _data_labels(self):
        values = self.df['data']
        labels = self.df['label'].squeeze()
        return values,labels 
    
    def __getitem__(self, index):
        # Get sample and label at index
        
        sample = self.values[index]
        label = torch.LongTensor([self.labels[index]]).squeeze()

        return sample, label
    
    def __len__(self):
        # Return the length of the dataset

        return len(self.labels)


@register('speck1_1_2e6_10')
class speck1(speck):
    # 生成数据格式为(2e6,64,10)
    # round = 1


    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck1_1_2e6_10.pkl')
        super().__init__(file_path)


@register('speck1_1_2e5_10')
class speck1(speck):


    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck1_1_2e5_10.pkl')
        super().__init__(file_path)


@register('speck1_1_2e4_10')
class speck1(speck):
    # 生成数据格式为(2e4,64,10)
    # round = 1


    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck1_1_2e4_10.pkl')
        super().__init__(file_path)



@register('speck2_1_2e7')
class speck1(speck):

    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck2_1_2e7.pkl')
        super().__init__(file_path)

@register('speck2_1_2e6')
class speck1(speck):

    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck2_1_2e6.pkl')
        super().__init__(file_path)



@register('speck2_1_2e5')
class speck1(speck):

    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/speck/speck2_1_2e5.pkl')
        super().__init__(file_path)
if __name__ == '__main__':
    a = make('speck1_1_2e4_10')

    print(len(a))
    print(a.labels.shape)
    print(a.values.shape)