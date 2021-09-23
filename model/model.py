from numpy.core.fromnumeric import shape

from torch.nn.modules.container import Sequential
import torchvision
import torch.nn as nn
import torch
import numpy as np

class VGG16_bn(nn.Module):
    def __init__(self, use_label):
        super().__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        # print(vgg16)
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.avgpool = nn.Sequential(*list(vgg16.avgpool.children()))
        self.classfication = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        self.use_label_num = len(use_label)
        self.class_score = nn.Linear(4096, self.use_label_num + 1)


    def forward(self, images, labels=None):
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classfication(x)
        cls_score = self.class_score(x)
        return cls_score


class ResNet34(nn.Module):
    def __init__(self, use_label):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        use_label_len = len(use_label)
        self.resnet.fc = nn.Linear(512, use_label_len+1)


    def forward(self, images, labels=None):
        cls_score = self.resnet(images)
        return cls_score

if __name__ == '__main__':
    vgg = VGG16_bn()
