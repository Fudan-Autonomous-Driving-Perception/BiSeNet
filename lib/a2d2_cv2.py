#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.a2d2_base_dataset import BaseDataset



class A2D2Data(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(A2D2Data, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 38
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )





if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = A2D2Data('/home/audituser/cyr/STDC-Seg-parkingslot/parkingSlot/planform', annpath=None, mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
