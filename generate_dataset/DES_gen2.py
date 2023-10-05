# 生成数据格式为（4，2），分别是明密文， 用4个16bit字表示64bit。各用（4，1）列表存储
# label为 0，1，2，3，4 表示在哪个子空间。





from Crypto.Cipher import DES
import random,tqdm
import torch
import numpy as np
from utils import *


def gen_dataset(num):
    space_num = 2
    data_dict = {"data":[],"label":[]}
    for idx in tqdm.tqdm(range(num)):
        rand_list56,rand_list64 = gen_random_key()
        key = list2bin(rand_list64)
        key56 = list2bin(rand_list56)
        list_data,bin_data = get_random_data()
        des = DES.new(key, mode=DES.MODE_ECB)

        bin_cdata = des.encrypt(bin_data)
        list_cdata = bin2list(bin_cdata)
        data_dict["data"].append([byte2word(list_data),byte2word(list_cdata)])
        data_dict["label"].append(key_idx(key56,space_num))
    
    data_dict["data"] = torch.tensor(data_dict["data"],dtype=torch.float32)
    data_dict["label"] = torch.tensor([data_dict["label"]],dtype=torch.int8)
    import pickle
    with open('statics/datasets/DES/data2.pkl', 'wb') as file:
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

    # word = byte2word(list_data)
    # print(list_data)
    # print(word)

    gen_dataset(int(2e6))
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





