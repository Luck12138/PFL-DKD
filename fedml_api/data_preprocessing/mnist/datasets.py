# -*- coding:utf-8 -*-
# @FileName  :datasets.py
# @Time      :2022/12/18 15:19
# @Author    :lucas
import numpy as np
import torch.utils.data as data
# from torchvision.datasets import MNIST
from  torchvision.datasets import FashionMNIST

class MNIST_truncated(data.Dataset):

    def __init__(self, root, cache_data_set=None,dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self,cache_data_set):
        # print("download = " + str(self.download))
        if cache_data_set == None:
            # mninst_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mninst_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        else:
            mninst_dataobj =  cache_data_set
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
        data = np.array(mninst_dataobj.data)
        target = np.array(mninst_dataobj.targets)


        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
