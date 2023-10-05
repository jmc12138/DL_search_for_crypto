


from Crypto.Cipher import DES
import random,tqdm
import torch
import numpy as np


def list2bin(_list):
    _list = [str(item) for item in _list]
    _str = ''.join(_list)
    _bin = int(_str, 2).to_bytes(8,'little')
    return _bin
def gen_random_key():
    rand_list56 = [random.randint(0, 1) for i in range(56)]
    rand_list64 = [0] * 64

    idx56 = 0
    for idx in range(64):

        if (idx+1) % 8 != 0:
            rand_list64[idx] = rand_list56[idx56]
            idx56 += 1 
    return rand_list56,rand_list64


def get_random_data():
    rand_list = [random.randint(0, 1) for i in range(64)]
    return rand_list,list2bin(rand_list)

def bin2list(_bin):
    _str = bin(int.from_bytes(_bin,'little'))[2:]

    _list = [int(item) for item in _str]

    _list = [0] * int(64 - len(_list)) + _list

    return _list



def key_idx(key,space_num):
    int_key = int.from_bytes(key,'little',signed=False)

    max_uint64 = np.uint64(2**56 - 1)


    len_space = max_uint64 // space_num
    idx = int(int_key // len_space)
    return idx


def byte2word(_list):
    ret = []
    step = 16
    for idx in range(0,len(_list),step):
        _list8 = _list[idx:idx+step]
        word = int.from_bytes(list2bin(_list8),'little',signed=False)
        ret.append(word)

    return ret