from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
'''
data = pd.read_table('celeba_label/identity_CelebA.txt', header=None, delim_whitespace=True)

print(data.head(10).values)
print(data.describe())
print(len(data))


i = 0
images = []
for fname, label in data.values:
    path = os.path.join("aa", fname)
    label = torch.from_numpy(np.array([label - 1]))
    item = (path, label)
    images.append(item)
    i += 1
    if i == 10:
        break

class CelebADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_table(csv_file, header=None, delim_whitespace=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        sample = {'image': image, 'label': self.labels.iloc[idx, 1]}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


a = torch.ones(3, 5)
b = 2 * torch.ones(3, 5)
c = torch.cat((a,b), 1)
print(c.size())
c = c.unsqueeze(2).unsqueeze(3)
print(c.size())

d = c.mean(dim=1, keepdim=True)
print((d*d).sum(dim = 1))

a = torch.ones(3, 3) * 2
b = torch.ones(3, 3) * 4

print(((a - b) ** 2).sum())
'''
'''
a = torch.ones(4,2) * 2
print(a.get_device())
'''

import torch.nn as nn
# x = torch.ones(1,3)
# print(x.size(1))

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
output.backward()