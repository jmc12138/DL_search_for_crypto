# 生成数据格式为(num,2,8,n)
# 用4个16bit字表示64bit。明密文用（8，1）列表存储
# label为 0，1，任务就是判断这两个密钥是否在一个子空间
# num = 2e5个数据 
# n为用相同密钥生成明密文对数目,会生成两个不同的密钥。




from Crypto.Cipher import DES
import random,tqdm
import torch
import numpy as np
from utils import *

def gen_dataset(num,n):
    space_num = 2
    data_dict = {"data":[],"label":[]}
    for _ in tqdm.tqdm(range(num)):
        data = []
        key_idx_list = []
        for i in range(2):
            rand_list56,rand_list64 = gen_random_key()
            key = list2bin(rand_list64)
            key56 = list2bin(rand_list56)
            des = DES.new(key, mode=DES.MODE_ECB)
            key_idx_list.append(key_idx(key56,space_num)) 
            for __ in range(n):

                list_data,bin_data = get_random_data()
                bin_cdata = des.encrypt(bin_data)
                list_cdata = bin2list(bin_cdata)
                data.append(byte2word(list_data)+byte2word(list_cdata))

        data_dict["data"].append(data)
        data_dict["label"].append(int(key_idx_list[0] == key_idx_list[1] ))
    
    data_dict["data"] = torch.tensor(data_dict["data"],dtype=torch.float32).reshape((num,2,8,n))
    # data_dict["data"] = torch.tensor(data_dict["data"],dtype=torch.float32)
    # print(torch.tensor(data_dict["data"],dtype=torch.float32).shape)
    data_dict["label"] = torch.tensor([data_dict["label"]],dtype=torch.int8)
    import pickle
    with open('statics/datasets/DES/data4.pkl', 'wb') as file:
        pickle.dump(data_dict, file)

    # return data_dict




# main 
if __name__ == '__main__':


    rand_list56,rand_list64 = gen_random_key()

    key = list2bin(rand_list64)
    list_data,data = get_random_data()

    des = DES.new(key, mode=DES.MODE_ECB)
    cdata = des.encrypt(data)
    data2 = des.decrypt(cdata)

    gen_dataset(int(2e5),10)
    # print(dd)



    # print(bin2list(data2))
    # print(list_data)
    # print(data)
    # print(bin(data))
    # print(type(data))
    # print(cdata)
    # print(data2)
    # print(key)
    # print(len(key))
    # print(rand_list64)
    # print(rand_list56)





