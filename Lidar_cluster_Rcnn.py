from numpy.core.numeric import NaN
from numpy.lib.type_check import imag
from cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from bbox_utils import cls_bbox
from config import Config as cfg
from model.model import VGG16_bn
from model.bbox_regressor import bbox_regressor
from bbox_utils import convert_xyxy_to_xywh
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
        self.epoch_standard = 0
        self.running_loss = 0.0
        self.final_loss_list = []
        self.cate_loss = self.cateLoss_initializer()
        self.lidar = LidarCluster()
        self.cls_bbox = cls_bbox(cfg.Train_set.use_label)
        self.label_num = len(cfg.Train_set.use_label)
        self.backbone = VGG16_bn(cfg.Train_set.use_label).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        params = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params, lr = lr_temp, weight_decay = weight_decay_
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=3,
                                                        gamma=0.1)
        now = time.localtime()
        self.criterion = nn.CrossEntropyLoss()
        self.pt_name = f"./result/vgg16_model{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour+9}_{now.tm_min}.pt"


    def toTensor(self, images, labels, device):
        img = torch.stack(images).to(device)
        label = torch.stack(labels)

        return img, label

    # def one_hot_endcoding(self, labels, device):
    #     labels = labels.type(torch.LongTensor)
    #     class_num = len(cfg.Train_set.use_label) + 1
    #     onehot = F.one_hot(labels, num_classes=class_num).to(device)
    #     return onehot

    def categoryLossViewer(self, preds, labels):
        preds = preds.to('cpu').numpy()
        labels = labels.to('cpu').numpy()
        for pred, label in zip(preds, labels):
            category = cfg.Train_set.index_to_label[str(label)]
            self.cate_loss[category] += 1
            if pred == label:
                self.cate_loss[category+'_correct'] += 1
        # print(self.cate_loss)

    def cateLoss_initializer(self):
        cate_loss = {}
        for category in cfg.Train_set.label_index.keys():
            cate_loss[category] = 1e-6
            cate_loss[category+'_correct'] = 0
        return cate_loss

    # def val_function(self, images, labels, device, cal):
    #     images, pred_bboxes, check = self.lidar(images, lidar, targets, cal)
    #     # print('check ? : ', check)
    #     images, labels, bbox_dataset, label_len = self.cls_bbox(images, pred_bboxes, targets, device)
    #     with torch.no_grad():
    #         val_loss = 0.0
    #
    #     val_loss = self.backbone(images, labels)
    #     print("[Val loss] : ", val_loss)

    # def forward(self, images, lidar, targets=None, cal=None, device=None, optimizer=None):
    def __call__(self, images, lidar, targets=None, cal=None, device=None):
        # images, pred_bboxes = self.lidar(images,lidar, targets, cal)
        images, pred_bboxes, check = self.lidar(images,lidar, targets, cal)
        images_, labels_, bbox_dataset, label_len = self.cls_bbox(images, pred_bboxes, targets, device)
        running_corrects = 0
        running_loss = 0.0
        data_len = 0
        if check > 3:
            select_region = Propose_region(images_, labels_, self.transform)
            if len(select_region) != 0:
                dataset = torch.utils.data.DataLoader(select_region, batch_size=8,
                                                        shuffle=True, num_workers=0,
                                                        collate_fn=collate_fn)

                data_size = len(dataset)
                # self.lr_scheduler.step()
                for epoch, (images, labels) in enumerate(dataset):
                    if label_len == 1 :
                        print('@@@@@@@@@@@@@ only one!')
                        continue
                    self.backbone.train()

                    #   Convert to Tensor Image
                    images, labels = self.toTensor(images, labels, device)
                    data_len += len(images)
                    # print('img;', images.shape)
                    #   Convert Tensor Type
                    #   for Cross Entropy Loss
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)
                    # if labels[0].to('cpu').numpy() != 0:
                    #     print('in@@@')
                    #     img = images.to('cpu').permute(0, 2, 3, 1)
                    #     print(img.shape)
                    #     plt.imshow(img[0])
                    #     plt.show()
                    self.optimizer.zero_grad()
                    cls_score = self.backbone(images)
                    cls_loss = self.criterion(cls_score, labels)
                    _, preds = torch.max(cls_score.data, 1)

                    running_corrects += torch.sum(preds == labels.data)
                    running_loss += cls_loss.item()
                    self.categoryLossViewer(preds, labels.data)
                    if not torch.isfinite(cls_loss):
                        print('WARNING: non-finite loss, ending training :  ', epoch)
                        exit(1)

                    if epoch % 3 == 0:
                        print('cls_loss : ', cls_loss, 'labels : ', labels)

                    self.running_loss += cls_loss.item()
                    if self.epoch_standard % 100 == 0:
                        print(f'@@@@[Training {self.epoch_standard} : {self.running_loss / 100}')
                        self.final_loss_list.append(self.running_loss)
                        self.running_loss = 0.0

                    # writer.add_scalar('Cls_Loss',cls_loss, epoch )
                    # cls_loss += cls_loss.item()
                    cls_loss.backward()
                    self.optimizer.step()
                    self.epoch_standard += 1
            torch.save(
                self.backbone
                , self.pt_name)
            if data_len != 0:
                print(f"loss : {running_loss / data_len}, Acc: {running_corrects / data_len}")
                print(f"Cate loss : \n",
                      f"Background : {self.cate_loss['Background_correct'] / self.cate_loss['Background']} \n"
                      f"Car : {self.cate_loss['Car_correct'] / self.cate_loss['Car']} \n"
                      f"Pedistrian : {self.cate_loss['Pedestrian_correct'] / self.cate_loss['Pedestrian']}  \n"
                      f"Truck : {self.cate_loss['Truck_correct'] / self.cate_loss['Truck']} \n"
                      f"Cyclist : {self.cate_loss['Cyclist_correct'] / self.cate_loss['Cyclist']}")

if __name__ == '__main__':
    kitti = kitti_set(cfg.SRCPATH, 'train')
    images, lidar, targets = kitti[0]          
    model = Lidar_cluster_Rcnn()
    model(images, lidar, targets)
    # print(img)