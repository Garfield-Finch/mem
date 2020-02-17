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
    def __init__(self, data_root, batch=2, workers=8, scale=512, transpose=None):
        self.data_root = data_root
        self.batch = batch
        self.workers = workers

        # ToAdd: images transform
        if transpose != None:
            self.t = transpose
        else:
            self.t = T.Compose([
                T.Resize((scale, scale)),
                T.ToTensor()
            ])
        self.tar_t = T.ToTensor()

    def data_load(self):
        train_set = iPERDataset(
            data_root=self.data_root,
            subset='train',
            transform=self.t,
            target_transform=self.tar_t
        )

        # val_set = IDRiDClsDataset(
        #     data_root=self.data_root,
        #     subset='val',
        #     transform=self.t)

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

        # val_loader = data.DataLoader(
        #     dataset=val_set,
        #     batch_size=self.batch,
        #     num_workers=self.workers,
        #     pin_memory=True
        # )

        # test_loader = torch.utils.data.DataLoader(
        #     dataset=test_set,
        #     batch_size=self.batch,
        #     num_workers=self.workers,
        #     pin_memory=True,
        #     shuffle=False
        # )

        # return train_loader, val_loader, test_loader
        return train_loader, train_loader


class iPERDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, subset, transform=None, target_transform=None):

        super(iPERDataset, self).__init__()

        self.data_root = data_root
        self.subset = subset

        # csv_file = os.path.join(data_root, 'groundtruths/IDRiD_{}.csv'.format(subset))
        # self.path_label_list = []
        # with open(csv_file) as f:
        #     for row in csv.reader(f):
        #         self.path_label_list.append(row)
        # self.path_label_list.pop(0)
        # if subset != 'train':
        #     subset = 'val'

        # load items from a txt file
        nm_file = os.path.join(data_root, f'{subset}.txt')
        self.lst_dir = []
        with open(nm_file, 'r') as f:
            self.lst_dir = f.readlines()

        # note that all elements of self.lst_dir ends with '\n'

        self.input_nm_list = []
        self.input_statistics = []  # contains the number of each sub dir of images, to obtain the P1 of each item P2
        for i in self.lst_dir:
            dir_img_root = os.path.join(self.data_root, 'images_HD', i[:-1])
            dir_img = os.listdir(dir_img_root)
            self.input_statistics.append(len(dir_img))
            dir_img.sort()
            for j in range(len(dir_img)):
                dir_img[j] = os.path.join(self.data_root, 'images_HD', i[:-1], dir_img[j])
            self.input_nm_list += dir_img

        for i in range(1, len(self.input_statistics)):
            self.input_statistics[i] += self.input_statistics[i - 1]
        # insert a zero at the beginning for latter usage
        self.input_statistics.insert(0, 0)

        self.t = transform
        self.tar_t = target_transform

    def _load_img_pose(self, item):
        P_name = self.input_nm_list[item]
        P_img_path = os.path.join(self.data_root, P_name)
        if not P_img_path.endswith('.jpg'):
            P_img_path = P_img_path + '.jpg'
        P_img = Image.open(P_img_path).convert('RGB')

        # for some images in images_HD named by such as `0000.jpg', the corresponding kpt image is named by `000.jpg'
        lst_nm_kpt_img = P_name.split("/")
        nm_kpt_img = lst_nm_kpt_img[-1][:-4]
        if int(nm_kpt_img) // 1000 < 1:
            nm_kpt_img = str(int(nm_kpt_img)).zfill(3)
        kpt_map_pth = '/'
        for i in range(len(lst_nm_kpt_img) - 1):
            kpt_map_pth = os.path.join(kpt_map_pth, lst_nm_kpt_img[i])
        kpt_map_pth = os.path.join(kpt_map_pth, nm_kpt_img)
        if not kpt_map_pth.endswith('.jpg'):
            kpt_map_pth = kpt_map_pth + '.jpg'
        kpt_map_pth = kpt_map_pth.replace('images_HD', 'kpt_imgs').replace('.jpg', '.npy')
        BP_img = np.load(kpt_map_pth)

        return P_img, BP_img, P_name

    def __getitem__(self, item):
        """
        P1: the source image
        BP1: the keypoints of the source image
        P2: the target image
        BP2: the keypoints of the target image
        """

        def _process_kpt(BP_img):
            BP = torch.from_numpy(BP_img).float()  # h, w, c
            BP = BP.transpose(2, 0)  # c,w,h
            BP = BP.transpose(2, 1)  # c,h,w

            return BP

        size_img = 256

        if item == 0:
            P1_item = 0
        else:
            for i in range(1, len(self.input_statistics)):
                if item < self.input_statistics[i]:
                    P1_item = self.input_statistics[i - 1]
                    break
        # P1_img, BP1_img, P1_name = self._load_img_pose(P1_item)
        P2_img, BP2_img, P2_name = self._load_img_pose(item)

        trans_resize = T.Compose([
            T.Resize((size_img, size_img)),
            T.ToTensor()
        ])
        # P1 = trans_resize(P1_img)  # the tensor containing source image
        P2 = trans_resize(P2_img)  # the tensor containing target image

        # to add data augmentation
        # if self.t is not None:
        #     imageOri = self.t(image)

        # transAug = T.Compose([
        #     T.Resize((224, 224)),
        #     T.RandomHorizontalFlip(),
        #     # T.RandomRotation(degrees=90),
        #     # T.ToTensor()
        # ])
        # imageOri = transAug(image)
        # import random
        # rangle = 3 * random.random()
        # rangle = rangle // 3
        # rangle *= 90
        # imageOri = imageOri.rotate(rangle)
        # imageOri = self.tar_t(imageOri)
        #
        # transRand = T.Compose([
        #     T.Resize((300, 300)),
        #     T.RandomCrop((224, 224)),
        #     T.ToTensor()
        # ])

        # imageRc = transRand(image)
        #
        # anchor_trans = T.Compose([
        #     T.Resize((224, 224)),
        #     T.ToTensor()
        # ])
        # imageOri = anchor_trans(image)

        # if self.tar_t is not None:
        #     target = self.tar_t(target)

        # return imageOut, target_DR, target_DME

        # BP1 = _process_kpt(BP1_img)
        BP2 = _process_kpt(BP2_img)

        # BP1 = BP1.clone().resize_([19, size_img, size_img])
        BP2 = BP2.clone().resize_([19, size_img, size_img])

        BP = BP2

        return BP, BP

        # return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
        #         'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        return len(self.input_nm_list)


def main():
    pass


if __name__ == '__main__':
    main()
