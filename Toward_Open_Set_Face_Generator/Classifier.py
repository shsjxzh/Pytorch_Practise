import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, num_classes=10177, pic_size=512 * 4 * 4, vector_length=4096):
        super(Classifier, self).__init__()
        self.features = models.vgg19_bn(pretrained=True).features
        #reduce the complexity of the network
        self.classifier = nn.Sequential(
            nn.Linear(pic_size, vector_length),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(vector_length, num_classes)
        )
        '''
        self.classifier = nn.Sequential(
            nn.Linear(pic_size, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, num_classes)
        )
        '''
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.classifier2(x)
        return output, x