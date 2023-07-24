from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import time
import json
import sys
import pandas as pd
from tqdm import tqdm

data_path = 'data/HPA'
subdir = ['train']
colors = ['red', 'blue', 'green', 'yellow']
ext = ['png']


def Cal_mean_std(datapath, subdir, colors, ext):

    sum = dict()
    sum_squared = dict()
    file_list = dict()
    num = 0
    for color in colors:
        sum.update({color: 0})
        sum_squared.update({color: 0})
        file_list.update({color: []})

    data_path = os.path.join(os.getcwd(), datapath)
    files = pd.read_csv(f'{data_path}/train.csv').to_numpy()[:, 0]
    num = files.shape[0]
    for file in files:
        for color in colors:
            for extension in ext:
                for sub in subdir:
                    file_path = os.path.join(data_path, sub, f'{file}_{color}.{extension}')
                    file_list[color].append(file_path)

    pxls = 0
    for i in tqdm(range(num)):
        for color in colors:
            filename = file_list[color][i]
            img = cv2.imread(filename, 0) / 255.0
            sum[color] += np.sum(img)
            sum_squared[color] += np.sum(img**2)
        pxls += img.size
                                
    d = dict(num_imgs=num)
    for color in colors:
        mean = sum[color] / pxls
        std =  np.sqrt(sum_squared[color] / pxls - mean**2)
        d.update({f'{color}_mean': mean, f'{color}_std': std})

    return d

if __name__ == '__main__':
    d = Cal_mean_std(datapath=data_path, subdir=subdir, colors=colors, ext=ext)
    outpath = os.path.join(data_path, 'color_info.json')
    with open(outpath, 'w') as f:
        json.dump(d, f)