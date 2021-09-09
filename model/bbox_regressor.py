from torch._C import device
import torchvision 
import torch.nn as nn
import torch


class bbox_regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # in_features = 256*6*6
        in_features = 4
        mid_features = 128
        out_features = 4
        self.model = self.get_model(device)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.Linear(mid_features, out_features)
            # nn.Relu()
        )
        # self.regressor = nn.Linear(in_features, out_features)
        # self.regressor.to(device)
        self.criterion = nn.MSELoss()

    def get_model(self, device):
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False
        
        model.to(device)

        return model


    def forward(self, bbox_dataset):
        # trains = bbox_dataset['Train_box'][0]
        # targets = bbox_dataset['Target_box'][0]
        # print(trains)
        # print('train:', trains)
        # print('targets:', targets.shape)
        # print(bbox_dataset)
        # outputs = self.regressor(trains)
        # print('outpus:', outputs)
        # loss = self.criterion(outputs, targets)

        return loss
