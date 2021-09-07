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

        self.class_score = nn.Linear(4096, self.use_label_num + 1 )
        self.cross_entropy = nn.CrossEntropyLoss().cuda()   


    def loss_function(self, predict, gt):
        cls_loss = self.cross_entropy(predict, gt)
        # print(cls_loss)
        return cls_loss

    def forward(self, images, labels=None):

        x = self.features(images)
        # print('x:', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classfication(x)
        cls_score = self.class_score(x)

        # target = torch.LongTensor([0]).to(labels.device)
        cls_loss = self.loss_function(cls_score, labels)
        # print('cls_loss : ', cls_loss)

            
            # cls_loss.backward()
            # opt.step()
            # opt.zero_grad
            # print(torch.max(cls_score))
            # print(cls_score)
        return cls_loss

if __name__ == '__main__':
    vgg = VGG16_bn()
