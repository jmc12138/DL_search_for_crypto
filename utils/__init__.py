

import os

from datetime import datetime
import os
import shutil
import time
import adabound,adamod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import confusion_matrix,f1_score
import shutil
_log_path = None
_device = None

def set_device():
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt',encoding='utf-8'):
    now = datetime.now()
    current_time = now.strftime("%Y:%m:%d:%H:%M:%S")
    str = f'{current_time}  :  {obj}'
    print(str)
    if _log_path is not None:
        if not os.path.exists(_log_path):
            os.mkdir(_log_path)
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(str, file=f)


def compute_n_params(model, return_str=True):

    
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, optimizer_name,lr_name,lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.

    _ = {'SGD':SGD,'Adam':Adam,'Adamax':torch.optim.Adamax,'RMSprop':torch.optim.RMSprop,'AdaBound':adabound.AdaBound,'AdaMod':adamod.AdaMod,'Adagrad':torch.optim.Adagrad}
       
    optimizer = _[optimizer_name](params,lr)

    if lr_name == 'MultiStepLR':
        lr_scheduler = MultiStepLR(optimizer, milestones)
    elif lr_name == 'None':
        lr_scheduler = None
    elif lr_name == 'ReduceLROnPlateau':
        pass
    return optimizer, lr_scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class F1Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((2, 2))
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.tpr = 0
        self.tnr = 0

    def update(self, confusion_matrix):
        self.confusion_matrix += confusion_matrix


    def cal(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel()

        self.accuracy = (tp + tn) / (tn + fp + fn + tp)
        self.precision = tp / (fp + tp)
        self.recall = tp / (fn + tp)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.tpr = tp / (tp + fp)
        self.tnr = tn / (tn + fn)
        self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp




def calc_weight(dataset):
    total_step = len(dataset)
    weight = [0, 0]
    for _, label in dataset:
        weight[int(label)] += 1

    weight[0], weight[1] = weight[1], weight[0]
    return torch.tensor(list(map(lambda v: v / total_step, weight)))


def positive_nums(dataset):
    sums = 0
    for _,j in dataset:
        sums += j
    return sums




def writer_scalars(a,b,mode,writer,epoch):
    for i,j in zip(a,b):
        writer.add_scalars(i, {mode: j}, epoch)






class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    sgpu = ''
    if isinstance(gpu,int):
        for i in range(gpu):
            sgpu += f",{i}"
        sgpu = sgpu[1:]
    else:
        sgpu = gpu
    print('set gpu:', sgpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = sgpu

def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def load_last_model_state(path, device, model, optimizer, scheduler):
 
    checkpoint = torch.load(path, map_location=device)
    

    
    epoch = checkpoint["epoch"]

    va = checkpoint["va"]

    model.load_state_dict(checkpoint["model_sd"])
    
    optimizer.load_state_dict(checkpoint["optimizer_sd"])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint["lr_sd"])

    return epoch,va,model,optimizer, scheduler




def get_device_msg():
    print('device msg: ',end='')
    print(torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))



def NCCL_log():
    import os 
    os.system('export NCCL_DEBUG=VERSION')
    # os.system('export NCCL_SOCKET_IFNAME=eth0')
    # os.system('export NCCL_IB_DISABLE=1')
    print("NCCL log 模式开启")

import datasets
from torch.utils.data import DataLoader


def get_dataLoader_weight(opt):
    dataset = datasets.make(opt.dataset_name)
    train_size = np.int32(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    positive_nums1 = positive_nums(train_dataset)
    positive_nums2 = positive_nums(test_dataset)
    log("数据集导入完毕")
    log(
        f"train dataset: len: {len(train_dataset)} shape:{train_dataset[0][0].shape},postive:{positive_nums1},negative:{len(train_dataset)-positive_nums1}"
    )
    log(
        f"test dataset: len: {len(test_dataset)} shape:{test_dataset[0][0].shape},postive:{positive_nums2},negative:{len(test_dataset)-positive_nums2}"
    )
    train_loader = DataLoader(
        train_dataset,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader,test_loader,calc_weight(train_dataset)