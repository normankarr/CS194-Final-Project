import torch
from torchvision import models
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.conv = nn.Conv2d(in_channels)
        self.model = nn.Sequential(*[
            nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, 35, kernel_size=(1, 1), stride=(1, 1))
        ])
        
    def forward(x):
        x = self.model(x)
        return x
    
# For Fine tuning
class fcn_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4].out_channels = num_classes

    def forward(self, x):
        x = self.model(x)
        return x

# Transfer Learning off of an imagenet resnet18
class fcn_resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained_net = models.resnet18(pretrained=True)
        modules = list(pretrained_net.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.model.add_module("conv", nn.Conv2d(512, num_classes, kernel_size=1))
        self.model.add_module("upsample", nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# Transfer Learning off of an imagenet resnet34
class fcn_resnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained_net = models.resnet34(pretrained=True)
        modules = list(pretrained_net.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.model.add_module("conv", nn.Conv2d(512, num_classes, kernel_size=1))
        self.model.add_module("upsample", nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    
    def forward(self, x):
        x = self.model(x)
        return x
    