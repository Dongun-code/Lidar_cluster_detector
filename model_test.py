import torchvision.transforms.functional

# from cluster_part import LidarCluster
from new_cluster_part import LidarCluster
from load_data.kitti_loader import kitti_set
from bbox_utils import cls_bbox

from model.model import VGG16_bn
from load_data.propose_region_test import Propose_region_test
from config import Config as cfg
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from model.nms import NMS
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import time
import copy
import cv2

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
        self.backbone = torch.load('./result/vgg16_model2021_9_10_0_20.pt').to(device)
        self.backbone.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params, lr=lr_temp, weight_decay=weight_decay_
        )

        now = time.localtime()
        self.pt_name = f"./result/vgg16_model{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour+9}_{now.tm_min}.pt"


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
        image_orig = copy.deepcopy(images)
        image_orig = np.array(image_orig)
        images_, pred_bboxes_, check = self.lidar(images, lidar, targets, cal)
        class_list = []
        bbox_list = []
        confidence_list = []
        # print('check ? : ', check)

        if len(images_) != 0:
            # images, labels, bbox_dataset, label_len = self.cls_bbox(images_, pred_bboxes_, targets, device)
            select_region = Propose_region_test(images_, pred_bboxes_, self.transform)
            if len(select_region) != 0:
                dataset = torch.utils.data.DataLoader(select_region, batch_size=1,
                                                      shuffle=False, num_workers=0,
                                                      collate_fn=collate_fn)

                for epoch, (images, bbox) in enumerate(dataset):
                # for show predict bbox
                #     for index in range(len(bbox)):
                #         # cls = bbox[index]
                #         # cls_name = cfg.Train_set.label_list[int(cls)]
                #         bboxx = bbox[index]
                #         bbox = bboxx[0]
                #         image_orig = cv2.rectangle(image_orig, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                # plt.imshow(image_orig)
                # plt.show()

                    # if label_len == 1:
                    #     print('@@@@@@@@@@@@@ only one!')
                    #     continue

                    # images, labels = self.toTensor(images, labels, device)
                    # labels = labels.type(torch.LongTensor)
                    # labels = labels.to(device)
                    image = images[0].to(device)
                    cls_score = self.backbone(image[None])
                    softmax_result = F.softmax(cls_score, dim=1)
                    # softmax_s = softmax_result.sum()
                    confidence, preds = torch.max(softmax_result.data, 1)
                    # print("Cls score : ", cls_score.to('cpu'))
                    # print("Predicted Class : ", preds)
                    # all_list.append(preds)

                    if preds != 0:
                        class_list.append(preds.to('cpu').numpy())
                        bbox_list.append(bbox)
                        confidence_list.append(confidence.to('cpu').numpy())
                # Need NMS
                if len(class_list) != 0:
                    final_box, final_class = NMS(bbox_list, class_list, confidence_list, 0.7, 0.5)

                    for index in range(len(final_box)):
                        cls = final_class[index]
                        cls_name = cfg.Train_set.label_list[int(cls)]
                        bbox = final_box[index]
                        bbox = bbox[0][0]
                        image_orig = cv2.rectangle(image_orig, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                        cv2.putText(image_orig, cls_name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


                plt.imshow(image_orig)
                plt.show()
                    # img = images.to('cpu')
                    # img = img[0].permute(1, 2, 0)
                    # # pil_img = to_pil_image(img)
                    # plt.imshow(img)
                    # plt.show()



if __name__ == '__main__':
    kitti = kitti_set(cfg.SRCPATH, 'train')
    images, lidar, targets = kitti[0]
    model = Lidar_cluster_Rcnn()
    model(images, lidar, targets)
    # print(img)