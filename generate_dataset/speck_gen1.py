
# 生成数据格式为(2e4,64,10)
# round = 1



import numpy as np
import speck,os
from os import urandom


def key_idx(key,space_num):


    max_uint64 = np.uint64(2**64 - 1)


    len_space = max_uint64 // space_num
    idx = int(key // len_space)
    return idx

def get_dataset(data_num,num_space,chunk_num,round):

    n = int(data_num)
    nr = int(round)

    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    keys64 = np.frombuffer(keys, dtype=np.uint64)
    ks = speck.expand_key(keys, nr)
    
    chunks = []
    for _ in range(chunk_num):
        plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
        ctdata0l, ctdata0r = speck.encrypt((plain0l, plain0r), ks)
        chunks.append([plain0l, plain0r,ctdata0l, ctdata0r])
    
    # data = np.array([speck.convert_to_binary(item) for item in chunks]).transpose(1,2,0)  
    # label = np.array([key_idx(key,num_space) for key in keys64])
    data = np.array([speck.convert_to_binary(item) for item in chunks],dtype=np.float32).transpose(1,0,2) 
    data_dict = {"data":data,
                 "label":np.array([key_idx(key,num_space) for key in keys64])}


    import pickle
    if not os.path.exists(r'statics/datasets/speck'):
        os.makedirs(r'statics/datasets/speck')
    with open(F'statics/datasets/speck/speck1_{int(round)}_{"{:.0e}".format(data_num).replace("+", "").replace("0", "")}_{int(chunk_num)}.pkl', 'wb') as file:
        pickle.dump(data_dict, file)


# main 
if __name__ == '__main__':

    data_num = 2e7
    num_space = 2
    chunk_num = 10
    round = 1
    get_dataset(data_num,num_space,chunk_num,round)



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





