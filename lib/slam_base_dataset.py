#!/usr/bin/python
# -*- encoding: utf-8 -*-
import pandas as pd
import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

from lib.sampler import RepeatedDistSampler



class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        self.data = pd.read_csv(dataroot + '/' + self.mode + '.csv', header=None, names=["image","label"])
        self.img_paths = self.data["image"].values[1:]
        self.lb_paths = self.data["label"].values[1:]

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx].replace('labels', 'masks')
        img, label = self.get_image(impth, lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def get_image(self, impth, lbpth):
        img, label = cv2.imread(impth)[:, :, ::-1], cv2.imread(lbpth, 0)
        return img, label

    def __len__(self):
        return self.len



