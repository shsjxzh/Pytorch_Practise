import torch
import torch.nn as nn

class my_vgg(nn.Module):
    # pic_size need to be caculated carefully
    def __init__(self, features, num_classes=1000, pic_size=512 * 4 * 4, hidden_node=4096, init=True):
        super(my_vgg, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(pic_size, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, num_classes)
        )
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

def my_vgg19_b(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = my_vgg(make_layers(cfg), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('c_params.pkl'))
    return model

# print(my_vgg19_b())