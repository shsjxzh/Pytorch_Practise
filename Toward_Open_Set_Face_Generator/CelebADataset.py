from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
'''
def make_dataset(dir, labels):
    images = []
    for fname, label in labels:
        path = os.path.join(dir, fname)
        item = (path, )
'''

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

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = pil_loader(img_name)
        # image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array([self.labels.iloc[idx, 1] - 1]))


'''
# The test of CelebADataset

face_data = CelebADataset(csv_file='celeba_label/identity_CelebA.txt',
                          root_dir='img_align_celeba_small',
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ])
                          )

fig = plt.figure()

for i in range(len(face_data)):
    sample = face_data[i]
    
    print(i, sample[0].shape, sample[1].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample[0].data.numpy().transpose((1,2,0)))

    if i == 3:
        plt.show()
        break
    if i == 3:
        break
'''