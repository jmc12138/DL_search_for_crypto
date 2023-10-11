
# 生成数据格式为(2e6,32), 分别代表样本量，数据，采用相同明文
# round = 1




import numpy as np
import speck,os
from os import urandom


def key_idx(key,space_num):


    max_uint64 = np.uint64(2**64 - 1)


    len_space = max_uint64 // space_num
    idx = int(key // len_space)
    return idx

def get_dataset(data_num,num_space,round):

    plainl = [0]  
    plainr = [0]  
    plain0l = np.array( [plainl] * int(data_num),dtype=np.uint16).reshape(-1)
    plain0r = np.array( [plainr] * int(data_num),dtype=np.uint16).reshape(-1)


    n = int(data_num)
    nr = int(round)

    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    keys64 = np.frombuffer(keys, dtype=np.uint64)
    ks = speck.expand_key(keys, nr)
    

    ctdata0l, ctdata0r = speck.encrypt((plain0l, plain0r), ks)

    # data = np.array([speck.convert_to_binary(item) for item in chunks]).transpose(1,2,0)  
    # label = np.array([key_idx(key,num_space) for key in keys64])
    data = np.array(speck.convert_to_binary32([ctdata0l, ctdata0r]) ,dtype=np.float32)
    data = data.reshape((-1,1,data.shape[-1])) 
    data_dict = {"data":data,
                 "label":np.array([key_idx(key,num_space) for key in keys64])}


    import pickle
    if not os.path.exists(r'statics/datasets/speck'
                          
                          ):
        os.makedirs(r'statics/datasets/speck')
    with open(F'statics/datasets/speck/speck2_{int(round)}_{"{:.0e}".format(data_num).replace("+", "").replace("0", "")}.pkl', 'wb') as file:
        pickle.dump(data_dict, file)


# main 
if __name__ == '__main__':

    data_num = 2e6
    num_space = 2
    round = 1
    get_dataset(data_num,num_space,round)



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





