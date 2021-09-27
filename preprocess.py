from new_cluster_part import LidarCluster
from new_box_util import cls_bbox
from config import Config as cfg
from torchvision import transforms
from load_data.proposal_region import Propose_region
from model.data_augmentation import data_augmentation

import torch


def collate_fn(batch):
    return zip(*batch)


class preProcess:
    def __init__(self):
        self.lidar = LidarCluster()
        self.cls_bbox = cls_bbox(cfg.Train_set.use_label)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def toTensor(self, images, labels, device):
        img = torch.stack(images).to(device)
        labels_tensor = torch.stack(labels)

        return img, labels_tensor

    def __call__(self, images, lidar, targets=None, cal=None, device=None):

        images = images[0]
        lidar = lidar[0]
        cal = cal[0]
        targets = {k: v.to(device) for k, v in targets[0].items()}

        images, pred_bboxes, check = self.lidar(images, lidar, targets, cal)
        images_, labels_, true_len = self.cls_bbox(images, pred_bboxes, targets, device)
        # images_, labels_ = data_augmentation(images_, labels_, device)

        select_region = Propose_region(images_, labels_, self.transform)
        if select_region != 0:
            dataset = torch.utils.data.DataLoader(select_region, batch_size=8,
                                                  shuffle=True, num_workers=0,
                                                  collate_fn=collate_fn)
            dataset_check = True
        elif len(select_region) == 0 or true_len == 0:
            dataset_check = False

        return dataset, dataset_check
