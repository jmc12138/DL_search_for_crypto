from datasets import make


import torch.utils.data as data
try:
    from .datasets import register
except:
    from datasets import register
import os,torch
import numpy as np
import pandas as pd

class cwe(data.Dataset):
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




@register('DES4_2e5')
class cwe119(cwe):
    # 生成数据格式为(num,2,8,n)
    # 用4个16bit字表示64bit。明密文用（8，1）列表存储
    # label为 0，1，任务就是判断这两个密钥是否在一个子空间
    # num = 2e5个数据 
    # n为用相同密钥生成明密文对数目,会生成两个不同的密钥。

    def __init__(self):
        file_path = os.path.abspath(r'statics/datasets/DES/data4_2e5.pkl')
        super().__init__(file_path)



if __name__ == '__main__':
    a = make('DES4_2e5')

    print(len(a))
    print(a.labels.shape)