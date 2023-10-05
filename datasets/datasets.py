import os
import pandas as pd
from glob import glob

datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    # print(datasets)

    dataset = datasets[name](**kwargs)
    return dataset



if __name__ == '__main__':
    make('DES',1)