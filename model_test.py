from cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from bbox_utils import cls_bbox

from model.model import VGG16_bn
from load_data.proposal_region import Propose_region
from config import Config as cfg
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import time


# writer = SummaryWriter('./result/log')

def collate_fn(batch):
    return zip(*batch)


class modelTest():
    def __init__(self, device, lr_temp, weight_decay_) -> None:
        super().__init__()
        print("is it start only one??")
        self.lidar = LidarCluster()
        self.cls_bbox = cls_bbox(cfg.Train_set.use_label)
        self.label_num = len(cfg.Train_set.use_label)
        self.backbone = VGG16_bn(cfg.Train_set.use_label).to(device)
        checkpoint = torch.load('./result/vgg16_model_bbox_regressor_2021_9_8_26_14')
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        # print(params)
        # print("parameter : ", params)
        self.optimizer = torch.optim.Adam(
            params, lr=lr_temp, weight_decay=weight_decay_
        )
        self.backbone.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        now = time.localtime()
        self.pt_name = f"./result/vgg16_model{now.tm_year}_{now.tm_mon}_{now.tm_hour}_{now.tm_min}"
        # model.train()
        # self.bbox_regressor = bbox_regressor().to(device)

    def toTensor(self, images, labels, device):
        img = torch.stack(images).to(device)
        label = torch.stack(labels)

        return img, label

    def one_hot_endcoding(self, labels, device):
        labels = labels.type(torch.LongTensor)
        class_num = len(cfg.Train_set.use_label) + 1
        onehot = F.one_hot(labels, num_classes=class_num).to(device)
        return onehot

    # def forward(self, images, lidar, targets=None, cal=None, device=None, optimizer=None):
    def __call__(self, images, lidar, targets=None, cal=None, device=None, mode='train'):
        # images, pred_bboxes = self.lidar(images,lidar, targets, cal)
        images, pred_bboxes, check = self.lidar(images, lidar, targets, cal)
        # print('check ? : ', check)

        if check > 3 or len(images) != 0:
            if mode == 'train':
                self.backbone.train()
            elif mode == 'test':
                self.backbone.eval()

            images, labels, bbox_dataset, label_len = self.cls_bbox(images, pred_bboxes, targets, device)
            select_region = Propose_region(images, labels, self.transform)
            if len(select_region) != 0:
                dataset = torch.utils.data.DataLoader(select_region, batch_size=6,
                                                      shuffle=True, num_workers=0,
                                                      collate_fn=collate_fn)
                # self.lr_scheduler.step()
                # print('lr_schedular:', )
                for epoch, (images, labels) in enumerate(dataset):
                    if label_len == 1:
                        print('@@@@@@@@@@@@@ only one!')
                        continue
                    self.backbone.eval()

                    images, labels = self.toTensor(images, labels, device)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)

                    cls_loss = self.backbone(images)





if __name__ == '__main__':
    kitti = kitti_set(cfg.SRCPATH, 'train')
    images, lidar, targets = kitti[0]
    model = Lidar_cluster_Rcnn()
    model(images, lidar, targets)
    # print(img)