import torch
import torch.nn as nn

class AttributeDecoder(nn.Module):
    # pic_size need to be caculated carefully
    def __init__(self, features, num_classes=1000, pic_size=512 * 4 * 4, hidden_node=4096):
        super(AttributeDecoder, self).__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def AttributeDecoder_19_b(**kwargs):
    model = AttributeDecoder(make_layers(cfg), **kwargs)
    return model

# print(AttributeDecoder19_b())