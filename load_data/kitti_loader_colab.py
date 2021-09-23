import os.path as op
import sys

# from numpy.lib.type_check import imag
sys.path.append("..")
from config import Config as cfg
from PIL import Image
from torchvision import transforms
from cluster_part import LidarCluster
from bbox_utils import cls_bbox
from kitti_util import Calibration
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

class kitti_set(torch.utils.data.Dataset):
    """
    input:
    path : data path
    split : train or test set

    return
    img : [N, height, width, 3]
    bbox : [N, [x1,y1,x2,y2]], [N,4]
    category : [one-hot-encoding], [N, 4]
    """
    def __init__(self, path, split) -> None:
        # self.VELOPATH = op.join(path, 'velo', split, 'velodyne')
        # self.IMGPATH = op.join(path,'img', split, 'image_2')
        # self.LABELPATH = op.join(path, 'label_2')
        self.VELOPATH = cfg.VELOPATH
        self.IMGPATH = cfg.IMGPATH
        self.LABELPATH = cfg.LABELPATH
        self.CALPATH = cfg.CALPATH
        self.use_label = cfg.Train_set.use_label
        self.file_list = self.check_file_num()
    # def label_preprocess(self, idx):
    #     pass

    def load_gt_bbox(self, path):
        with open(path, 'r') as r:
            anns = r.readlines()  
        bboxes = []
        category = []
        for ann in anns:
            ann = ann.strip('\n').split(' ')
            # print(ann[0])
            if ann[0] in self.use_label:
                index = cfg.Train_set.label_index[ann[0]]
                category.append(index)
                bboxes.append([float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])])
        # print(bbox)
        return category, bboxes

    def check_file_num(self):
        path = op.join(self.LABELPATH, '*.txt')
        file = glob.glob(path)
        return file

    def __getitem__(self, idx):
        label_file = '{0:06d}.txt' .format(idx)
        lidar_file = '{0:06d}.bin' .format(idx)
        img_file = '{0:06d}.png' .format(idx)
        cal_file = '{0:06d}.txt' .format(idx)

        category, bboxes = self.load_gt_bbox(op.join(self.LABELPATH, label_file))
        points = np.fromfile(op.join(cfg.VELOPATH, lidar_file), dtype=np.float32).reshape(-1, 4)       
        intensity = points[:, 3]
        points = points[:, 0:3]
        lidar = dict(points=points, intensity=intensity)

        cal = Calibration(op.join(self.CALPATH, cal_file))

        img = Image.open(op.join(cfg.IMGPATH, img_file)) 

        # img = transforms.ToTensor()(img)
     
        # self.img_num = len(img)
        bboxes = torch.tensor(bboxes)
        category = torch.tensor(category, dtype=torch.float32)
        targets = dict(bboxes=bboxes, category=category)

        return img, lidar, targets, cal


    def __len__(self):
        return len(self.file_list)

# if __name__ == '__main__':
#     # LABELPATH = op.join(cfg.SRCPATH, 'label_2')
#     # cls = cls_bbox(LABELPATH)
#     # cls(1)
#     kitti = kitti_set(cfg.SRCPATH, 'train')
#     img, lidar, targets = kitti[0]          
#     # print(img)