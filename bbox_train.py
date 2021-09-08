from numpy.core.numeric import NaN
from numpy.lib.type_check import imag
from cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from bbox_utils import cls_bbox
from config import Config as cfg
from model.model import VGG16_bn
from load_data.proposal_region import Propose_region
from config import Config as cfg
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model.bbox_regressor import bboxRegressor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import time
writer = SummaryWriter('./result/log')


def collate_fn(batch):
    return zip(*batch)


class trainBBox:
    def __init__(self, device, lr_temp, weight_decay_):
        super().__init__()
        print("BBox Regressor Train")
        self.epoch_standard = 0
        self.running_loss = 0.0
        self.final_loss_list = []
        self.lidar = LidarCluster()
        self.cls_bbox = cls_bbox(cfg.Train_set.use_label)
        self.label_num = len(cfg.Train_set.use_label)
        # self.backbone = VGG16_bn(cfg.Train_set.use_label).to(device)
        self.model = bboxRegressor(device).to(device)
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        params = [p for p in self.model.parameters() if p.requires_grad]
        # print(params)
        # print("parameter : ", params)
        self.optimizer = torch.optim.Adam(
            params, lr=lr_temp, weight_decay=weight_decay_
        )
        # optimizer = torch.optim.Adadelta(
        #     params, lr=lr_temp, weight_decay= weight_decay_
        # )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=3,
                                                            gamma=0.1)

        now = time.localtime()
        self.pt_name = f"./result/vgg16_model_bbox_regressor_{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour+9}_{now.tm_min}"


    def toTensor(self, images, labels, target_bbox, device):
        img = torch.stack(images).to(device)
        label = torch.stack(labels)
        target_bbox = torch.stack(target_bbox)
        use_index = label > 0

        out_img = img[use_index]
        out_label = label[use_index]
        out_target_bbox = target_bbox[use_index]

        return out_img, out_label, out_target_bbox

    def one_hot_endcoding(self, labels, device):
        labels = labels.type(torch.LongTensor)
        class_num = len(cfg.Train_set.use_label) + 1
        onehot = F.one_hot(labels, num_classes=class_num).to(device)
        return onehot

    # def forward(self, images, lidar, targets=None, cal=None, device=None, optimizer=None):
    def __call__(self, images, lidar, targets=None, cal=None, device=None):
        # images, pred_bboxes = self.lidar(images,lidar, targets, cal)
        images, pred_bboxes, check = self.lidar(images, lidar, targets, cal)
        images, labels, target_bbox, label_len = self.cls_bbox(images, pred_bboxes, targets, device)
        epoch_standard = 0
        running_loss = 0.0

        if check > 3:
            select_region = Propose_region(images, labels, target_bbox, self.transform)
            dataset = torch.utils.data.DataLoader(select_region, batch_size=6,
                                                  shuffle=True, num_workers=0,
                                                  collate_fn=collate_fn)
            # self.lr_scheduler.step()
            # print('lr_schedular:', )
            for epoch, (images, labels, target_bbox) in enumerate(dataset):
                if label_len == 1:
                    # print('@@@@@@@@@@@@@ only one!')
                    continue
                self.model.train()

                images, labels, target_bboxes = self.toTensor(images, labels, target_bbox, device)
                if len(target_bboxes) == 0:
                    
                    print("BBoxes is None")
                    continue

                bbox_loss = self.model(images, target_bboxes)

                if not torch.isfinite(bbox_loss):
                    print('WARNING: non-finite loss, ending training :  ', epoch)
                    exit(1)

                if epoch % 3 == 0:
                    print('cls_loss : ', bbox_loss, 'labels : ', labels)

                self.running_loss += bbox_loss.item()
                if self.epoch_standard % 100 == 0:
                    print(f'@@@@[Training {self.epoch_standard} : {self.running_loss / 100}')
                    self.final_loss_list.append(self.running_loss)
                    self.running_loss = 0.0

                #  writer.add_scalar('bbox_Loss', bbox_loss, epoch )
                # cls_loss += cls_loss.item()

                self.optimizer.zero_grad
                bbox_loss.backward()
                self.optimizer.step()

                self.epoch_standard += 1

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': bbox_loss,
                    'epoch': epoch,
                },  self.pt_name)



if __name__ == '__main__':
    kitti = kitti_set(cfg.SRCPATH, 'train')
    images, lidar, targets = kitti[0]
    model = Lidar_cluster_Rcnn()
    model(images, lidar, targets)
    # print(img)