import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.utils.data as Data
# import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Hyper parameter:
EPOCHS = 6                      # the training epochs
BATCH_SIZE = 64                 # not use all data to train
LR = 0.001
DOWNLOAD = True                 # if have already download, then turn it to 'False'
SHOW_STEP = 500                 # show the result after how many steps
PRETRAINED = False              # use pretrained model parameters
MODEL = 'vgg19_bn'              # use which model

# Data path:
traindir = 'train'
valdir = 'val'

def adjust_learning_rate(optimizer, epoch):
    # set the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    model = models.__dict__[MODEL](pretrained=PRETRAINED)
    if MODEL.startswith('alexnet') or MODEL.startswith('vgg'):
        model.features = nn.DataParallel(model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
    else:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        root=traindir,
        transforms=transforms.Compose([
            transforms.RandomResizedCrop(224),  # what is this?
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = Data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    val_loader = Data.DataLoader(

    )



if __name__ == '__main__':
    main()
