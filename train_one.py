import torch
from load_data.kitti_loader_val import kitti_set
from model.utils import collate_fn
from config import Config as cfg
from preprocess import preProcess
from load_data.kitti_loader_val import kitti_set as kitti_val

import torch.nn as nn
import torch.nn.functional as F

def toTensor(images, labels, device):
    img = torch.stack(images).to(device)
    labels_tensor = torch.stack(labels)

    return img, labels_tensor

def train_one_epoch(model, data_loader, optimizer, device, epoch, save_name):
    model.train()
    preprocess = preProcess()
    cate_loss = cateLoss_initializer()

    for i, (images, lidar, targets, cal) in enumerate(data_loader):
        # if i < 1000:
        #     continue
        print("[Data] : ", i)
        total_loss = []
        loss_sum = 0
        train_dataset, check = preprocess(images, lidar, targets, cal, device)
        if len(check) == False:
            continue

        for (images, labels) in train_dataset:
            images, labels = toTensor(images, labels, device)
            cls_score = model(images)

            _, preds = torch.max(cls_score.data, 1)
            categoryLossViewer(preds, labels.data, cate_loss, i)

            cls_loss = F.cross_entropy(cls_score, labels)
            total_loss.append(cls_loss)
            loss_sum += cls_loss
            if not torch.isfinite(cls_loss):
                print('WARNING: non-finite loss, ending training :  ', epoch)
                exit(1)

        print(f"[Epoch [{epoch}, i]] : {loss_sum / len(train_dataset)}")
        total_loss = torch.sum(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        torch.save(model, save_name)


def model_validation(self, device):
    print("[Model Validation Mode]")
    dataset = kitti_val(cfg.SRCPATH, 'val')
    d_val = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             collate_fn=collate_fn)
    self.backbone.eval()
    running_loss = 0
    data_num = len(self.d_val)
    preprocess = preProcess()

    for i, (images, lidar, targets, cal, img_file) in enumerate(d_val):

        val_dataset, check = preprocess(images, lidar, targets, cal, device)
        if len(check) == False:
            continue

        for (images, labels) in enumerate(val_dataset):
            images, labels = self.toTensor(images, labels, device)

            with torch.no_grad():
                cls_score = self.backbone(images)
                cls_loss = self.criterion(cls_score, labels)
                running_loss += cls_loss.item()

    mean_loss = running_loss / data_num
    print(f"[Model Val loss] : {mean_loss}")


def cateLoss_initializer():
    cate_loss = {}
    for category in cfg.Train_set.label_index.keys():
        cate_loss[category] = 1e-6
        cate_loss[category+'_correct'] = 0
    return cate_loss

def categoryLossViewer(preds, labels, cate_loss, index):
    preds = preds.to('cpu').numpy()
    labels = labels.to('cpu').numpy()
    for pred, label in zip(preds, labels):
        category = cfg.Train_set.index_to_label[str(label)]
        cate_loss[category] += 1
        if pred == label:
            cate_loss[category+'_correct'] += 1

    if index % 500 == 0:
        print(f"Cate loss : \n",
              f"Background : {cate_loss['Background_correct'] / cate_loss['Background']} \n"
              f"Car : {cate_loss['Car_correct'] / cate_loss['Car']} \n"
              f"Pedistrian : {cate_loss['Pedestrian_correct'] /cate_loss['Pedestrian']}  \n"
              f"Truck : {cate_loss['Truck_correct'] / cate_loss['Truck']} \n")

