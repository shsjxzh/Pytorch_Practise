import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision          # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision import models

# Hyper Parameters
EPOCH = 6                       # the training times
BATCH_SIZE = 64                 # not use all data to train
LR = 0.001
DOWNLOAD_MNIST = False          # if have already download, then turn it to 'False'
SHOW_STEP = 100                 # show the result after how many steps
PRETRAINED = False              # use pretrained model parameters
MODEL = 'vgg19_bn'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = models.__dict__[MODEL](pretrained=PRETRAINED, num_classes=10)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg_feature = nn.Sequential(*list(models.vgg19_bn().features))
        self.vgg_classifier = nn.Sequential(*list(models.vgg19_bn(num_classes=10).classifier.children()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.vgg_feature(x)
        x = self.vgg_classifier(x)
        return x


net = CNN()
print(net)
