from numpy.core.numeric import NaN
from numpy.lib.type_check import imag
# from cluster_part import LidarCluster
from new_cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from load_data.kitti_loader_val import kitti_set as kitti_val
# from bbox_utils import cls_bbox
from new_box_util import cls_bbox
from config import Config as cfg
from model.model import VGG16_bn, ResNet34
from model.bbox_regressor import bbox_regressor
from bbox_utils import convert_xyxy_to_xywh
from load_data.proposal_region import Propose_region
from save_param import write_options
from config import Config as cfg
from torchvision import transforms
from model.data_augmentation import data_augmentation
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

class Lidar_cluster_Rcnn(nn.Module):
    def __init__(self, device, lr_temp, weight_decay_, end_epoch) -> None:
        super().__init__()
        print("First Model Train Start")
        dataset = kitti_val(cfg.SRCPATH, 'val')
        self.d_val = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            shuffle=False, num_workers=0,
                                            collate_fn=collate_fn)

        self.backbone = VGG16_bn(cfg.Train_set.use_label).to(device)


    def forward(self, images):
        features = self.backbone(images)
        return features


#
# if __name__ == '__main__':
#     kitti = kitti_set(cfg.SRCPATH, 'train')
#     images, lidar, targets = kitti[0]
#     model = Lidar_cluster_Rcnn()
#     model(images, lidar, targets)
#     # print(img)