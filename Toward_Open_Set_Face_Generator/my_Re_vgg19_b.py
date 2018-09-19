import torch
import torch.nn as nn

class my_Re_vgg(nn.Module):
    # pic_size need to be caculated carefully
    def __init__(self, features):
        super(my_Re_vgg, self).__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 512 * 4 * 4 * 2
    for v in reversed(cfg):
        if v == 'M':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]   # there exist other mode, may be 'bilinear' is more suitable
        else:
            convt2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
            layers += [convt2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = [3, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def my_Re_vgg19_b(pretrained=False, **kwargs):
    model = my_Re_vgg(make_layers(cfg), **kwargs)
    if pretrained:
        pretrained_dict = torch.load('g_params.pkl',  map_location=lambda storage, loc: storage)
        model_dict=model.state_dict()
        
        pretrained_dict = {k.replace('module.','') : v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(torch.load('c_params.pkl',  map_location=lambda storage, loc: storage))
    return model


# print(my_Re_vgg19_b())