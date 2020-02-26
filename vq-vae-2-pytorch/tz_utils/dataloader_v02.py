# -*- coding: utf-8 -*-

import os
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as T
# from torchvision.datasets.folder import *

import numpy as np
import pickle


class iPERLoader:

    def __init__(self, data_root, batch=2, workers=8, img_size=256, transpose=None):
        self.data_root = data_root
        self.batch = batch
        self.workers = workers
        self.img_size = img_size

        # ToAdd: images transform
        if transpose != None:
            self.t = transpose
        else:
            self.t = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor()
            ])

    def data_load(self):
        train_set = iPERDataset(
            data_root=self.data_root,
            subset='train',
            transform=self.t,
        )

        val_set = iPERDataset(
            data_root=self.data_root,
            subset='val',
            transform=self.t,
        )

        # test_set = iPERDataset(
        #     data_root=self.data_root,
        #     subset='train',
        #     transform=self.t,
        #     target_transform=self.tar_t
        # )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True
        )

        # test_loader = torch.utils.data.DataLoader(
        #     dataset=test_set,
        #     batch_size=self.batch,
        #     num_workers=self.workers,
        #     pin_memory=True,
        #     shuffle=False
        # )

        # return train_loader, val_loader, test_loader
        return train_loader, val_loader


class iPERDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, subset, transform=None):

        super(iPERDataset, self).__init__()

        self.data_root = data_root
        self.subset = subset

        # load items from a txt file
        nm_file = os.path.join(data_root, f'{subset}.txt')
        self.lst_dir = []
        with open(nm_file, 'r') as f:
            self.lst_dir = f.readlines()

        # NOTE that all elements of self.lst_dir ends with '\n'
        # NOTE that the train.txt file should end with only ONE SINGLE blank line

        self.input_nm_list = []
        for i in self.lst_dir:
            dir_img_root = os.path.join(self.data_root, 'images_HD', i[:-1])
            dir_img = os.listdir(dir_img_root)
            dir_img.sort()
            for j in range(len(dir_img)):
                dir_img[j] = os.path.join(self.data_root, 'images_HD', i[:-1], dir_img[j])
            self.input_nm_list += dir_img

        self.transform = transform

    def __getitem__(self, item):
        img_app_nm = self.input_nm_list[item]
        img_pose_nm = img_app_nm.replace('images_HD', 'skeletons').replace('.jpg', '.png')
        img_app = Image.open(img_app_nm)
        img_pose = Image.open(img_pose_nm)

        tsr_app = self.transform(img_app)  # the tensor containing target image
        tsr_pose = self.transform(img_pose)

        return tsr_app, tsr_pose

    def __len__(self):
        return len(self.input_nm_list)


def main():
    pass


if __name__ == '__main__':
    main()
