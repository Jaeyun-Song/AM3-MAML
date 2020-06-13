import h5py
import json
import os
import io

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

# from utils import get_transform

class PretrainDataset(Dataset):
    def __init__(self, datapath, dataset):
        self.dataset = dataset
        if dataset == 'miniimagenet' or dataset == 'cub':
            self.datapath = os.path.join(datapath, dataset, 'train_data.hdf5')
            self.labelpath = os.path.join(datapath, dataset, 'train_labels.json')
            self.data = h5py.File(self.datapath, 'r', swmr=True)['datasets']
            self.label = json.load(open(self.labelpath,'r'))
            # slice data by self.data[self.label[i]]
            self.label_len = [len(self.data[i]) for i in self.label]
        elif dataset == 'fc100':
            self.datapath = os.path.join(datapath, 'cifar100', 'data.hdf5')
            self.labelpath = os.path.join(datapath, 'cifar100', 'fc100', 'train_labels.json')
            self.data = h5py.File(self.datapath, 'r', swmr=True)
            self.label = json.load(open(self.labelpath,'r'))
            # slice data by self.data[self.label[i][0]][self.label[i][1]]
            self.label_len = [len(self.data[i[0]][i[1]]) for i in self.label]
        elif dataset == 'cifarfs':
            self.datapath = os.path.join(datapath, 'cifar100', 'data.hdf5')
            self.labelpath = os.path.join(datapath, 'cifar100', 'cifar-fs', 'train_labels.json')
            self.data = h5py.File(self.datapath, 'r', swmr=True)
            self.label = json.load(open(self.labelpath,'r'))
            # slice data by self.data[self.label[i][0]][self.label[i][1]]
            self.label_len = [len(self.data[i[0]][i[1]]) for i in self.label]
        elif dataset == 'omniglot':
            self.datapath = os.path.join(datapath, dataset, 'data.hdf5')
            self.labelpath = os.path.join(datapath, dataset, 'train_labels.json')
            self.data = h5py.File(self.datapath, 'r', swmr=True)
            self.label = json.load(open(self.labelpath, 'r'))
            # slice data by self.data[self.label[i][0]][self.label[i][1]][self.label[i][2]]
            self.label_len = [len(self.data[i[0]][i[1]][i[2]]) for i in self.label]

    def __getitem__(self, i):
        class_id, i = self.int_to_id(i)
        if self.dataset == 'miniimagenet' or self.dataset == 'cub':
            return self.data[self.label[class_id]][i], class_id
        elif self.dataset == 'fc100' or self.dataset == 'cifarfs':
            return self.data[self.label[class_id][0]][self.label[class_id][1]][i], class_id
        elif self.dataset == 'omniglot':
            return self.data[self.label[class_id][0]][self.label[class_id][1]][self.label[class_id][2]][i], class_id

    def int_to_id(self, i):
        class_id = 0
        for num in self.label_len:
            if i >= num:
                class_id += 1
                i -= num
            else:
                return class_id, i
    
    def __len__(self):
        return int(np.sum(self.label_len))

class TransformDataset(Dataset):
    def __init__(self, dataset, transfrom=None):
        self.dataset = dataset
        # self.transform = transforms.Compose(transfrom)
        self.transform = transfrom

    def __getitem__(self, i):
        x,y = self.dataset[i]
        if self.transform:
            if self.dataset.dataset.dataset=='cub':
                x = self.transform(Image.open(io.BytesIO(x)).convert('RGB'))
            else:
                x = self.transform(Image.fromarray(x))
        return x,y

    def __len__(self):
        return len(self.dataset)

if __name__=='__main__':
    dataset = 'cub'
    _, train_transform, test_transform, feature_size = get_transform(dataset)
    dataset = PretrainDataset('../data',dataset)
    train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_set = TransformDataset(train_set, train_transform)
    val_set = TransformDataset(val_set, test_transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    for (img, label) in train_loader:
        print(img.shape, label.shape)