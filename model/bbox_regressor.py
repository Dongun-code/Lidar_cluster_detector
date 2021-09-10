import torchvision
import torch.nn as nn
import torch


class bboxRegressor(nn.Module):
    def __init__(self,   device):
        super().__init__()
        in_features = 512*7*7
        # in_features = 4
        mid_features = 128
        out_features = 4
        self.device = device
        self.model = self.get_model(device)
        # self.regressor = nn.Sequential(
        #     nn.Linear(in_features, mid_features),
        #     nn.Linear(mid_features, out_features)
        #     # nn.Relu()
        # )
        self.regressor = nn.Linear(in_features, out_features)
        self.regressor.to(device)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.criterion = nn.MSELoss()

    def get_model(self, device):
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.eval()
        # model = torch.load('./result/vgg16_model2021_9_18_19')

        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)

        return model


    def forward(self, images, target_bbox):

        features = self.model.features(images)
        x = torch.flatten(features, 1)
        pred = self.regressor(x)
        loss = self.criterion(pred, target_bbox)

        return loss
