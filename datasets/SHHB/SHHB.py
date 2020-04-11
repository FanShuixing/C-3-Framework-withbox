import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd
import re

from config import cfg
import json
from datasets.mat_to_npy import get_density


class SHHB(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.root_dir=data_path
        if mode == 'train':
            self.gt_csv=data_path+'train_meta.csv'
        else:
            self.gt_csv=data_path+'val_meta.csv'
        with open(self.gt_csv) as fr:
            self.data_files=pd.read_csv(fr).values

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den,wh,ind,mask = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den,wh,ind,mask

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.root_dir, fname[1]))
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img.resize((768, 576))
        den=get_density(os.path.join(self.root_dir, fname[1]),os.path.join(self.root_dir, fname[0]),w=768,h=576)

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        #wh
        self.max_objs=150
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        with open(os.path.join(self.root_dir,fname[0])) as fr:
            info=json.load(fr)
            num_box=info['num_box']
            img_w,img_h=768,576
            for k in range(num_box):
                x0=info['bboxes'][k]['x_min']*img_w
                x1=info['bboxes'][k]['x_max']*img_w
                y0=info['bboxes'][k]['y_min']*img_h
                y1=info['bboxes'][k]['y_max']*img_h
                w=(x1-x0)
                h=(y1-y0)

                wh[k] = 1. * w, 1. * h
                #ind
                output_w=768
                ct = np.array(
                  [(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                ind[k] = ct_int[1] * output_w + ct_int[0]
                if ind[k]>576*768:
                    print(ct_int[1],ct_int[0])
                    print('*'*100,'error')
                #reg_mask
                reg_mask[k] = 1

        return img, den,wh,ind,reg_mask

    def get_num_samples(self):
        return self.num_samples