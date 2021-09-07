from numpy.core.numeric import NaN
from numpy.lib.type_check import imag
from cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from bbox_utils import cls_bbox
from config import Config as cfg
from model.model import VGG16_bn
from model.bbox_regressor import bboxRegressor

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

class Lidar_cluster_Rcnn_continue():
    def __init__(self, device, lr_temp, weight_decay_) -> None:
        super().__init__()
        print("is it start only one??")
        self.lidar = LidarCluster()
        self.cls_bbox = cls_bbox(cfg.Train_set.use_label)
        self.label_num = len(cfg.Train_set.use_label)
        self.backbone = VGG16_bn(cfg.Train_set.use_label).to(device)
        checkpoint = torch.load('./result/vgg16_model2021_9_18_19')
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        # print(params)
        # print("parameter : ", params)
        self.optimizer = torch.optim.Adam(
            params, lr=lr_temp, weight_decay= weight_decay_
        )
        self.backbone.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer = torch.optim.Adadelta(
        #     params, lr=lr_temp, weight_decay= weight_decay_
        # )    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=3,
                                                        gamma=0.1)
        print('check epoch:', checkpoint['epoch'])
        now = time.localtime()
        self.pt_name = f"./result/vgg16_model{now.tm_year}_{now.tm_mon}_{now.tm_hour}_{now.tm_min}"
        # model.train()        
        # self.bbox_regressor = bbox_regressor().to(device)

    def toTensor(self, images,labels, device):
        img = torch.stack(images).to(device)
        label = torch.stack(labels)

        return img, label

    def one_hot_endcoding(self, labels, device):
        labels = labels.type(torch.LongTensor)
        class_num = len(cfg.Train_set.use_label) + 1
        onehot = F.one_hot(labels, num_classes=class_num).to(device)
        return onehot

    # def forward(self, images, lidar, targets=None, cal=None, device=None, optimizer=None):
    def __call__(self, images, lidar, targets=None, cal=None, device=None):
        # images, pred_bboxes = self.lidar(images,lidar, targets, cal)
        images, pred_bboxes, check = self.lidar(images,lidar, targets, cal)
        # print('check ? : ', check)
        images, labels, bbox_dataset, label_len = self.cls_bbox(images, pred_bboxes, targets, device)

        
        if check > 3:
            select_region = Propose_region(images, labels, self.transform)
            if len(select_region) != 0:
                dataset = torch.utils.data.DataLoader(select_region, batch_size=6,
                                                        shuffle=True, num_workers=0,
                                                        collate_fn=collate_fn)
                # self.lr_scheduler.step()
                # print('lr_schedular:', )
                for epoch, (images, labels) in enumerate(dataset):
                    if label_len == 1 :
                        print('@@@@@@@@@@@@@ only one!')
                        continue
                    self.backbone.train()
                    # print('original label:', labels)
                    #   Convert to Tensor Image
                    images, labels = self.toTensor(images, labels, device)
                    # print('img;', images.shape)
                    #   Convert Tensor Type
                    #   for Cross Entropy Loss
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)
                    # print('@@@@@@@labels', labels)
                    # if labels.to('cpu').numpy() != 0:
                    #     print('in@@@')
                    #     img = images.to('cpu').permute(0, 2, 3, 1)
                    #     print(img.shape)
                    #     plt.imshow(img[0])
                    #     plt.show()
                    cls_loss = self.backbone(images, labels)

                    if not torch.isfinite(cls_loss):
                        print('WARNING: non-finite loss, ending training :  ',epoch)
                        exit(1)

                    if epoch % 3 == 0:
                        print('cls_loss : ', cls_loss, 'labels : ', labels)

                    self.optimizer.zero_grad
                    cls_loss.backward()
                    self.optimizer.step()

                    torch.save({
                        'model_state_dict': self.backbone.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': cls_loss,
                        'epoch': epoch,
                        },self.pt_name)


if __name__ == '__main__':
    kitti = kitti_set(cfg.SRCPATH, 'train')
    images, lidar, targets = kitti[0]          
    model = Lidar_cluster_Rcnn()
    model(images, lidar, targets)
    # print(img)