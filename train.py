from model.utils import collate_fn
from torch.nn import parameter
# from model.maskrcnn import MaskRCNN
# from load_data.coco import coco_set
from config import Config as cfg
# from model.detection.engine import train_one_epoch, evaluate
from Lidar_cluster_Rcnn import Lidar_cluster_Rcnn
from model_test import modelTest
from bbox_train import trainBBox
import torch
import numpy as np
# from load_data.kitti_loader import kitti_set
from load_data.kitti_loader_colab import kitti_set
from train_one import train_one_epoch
from model.utils import collate_fn 
from train_continue import Lidar_cluster_Rcnn_continue
# from train_one import train_one_epoch
# from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
import gc
import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
gc.collect()
torch.cuda.empty_cache()
# from torchvision.references.detection import engine
writer = SummaryWriter()
def get_gpu_prop(show=False):

    ngpus = torch.cuda.device_count()
    
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_momory": round(prop.total_memory / 1073741824, 2), # unit GB
            "sm_count": prop.multi_processor_count
        })
       
    if show:
        print("cuda: {}".format(torch.cuda.is_available()))
        print("available GPU(s): {}".format(ngpus))
        for i, p in enumerate(properties):
            print("{}: {}".format(i, p))
    return properties


def main():
    device = torch.device("cuda")
    if device.type == "cuda": 
        get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    momentum = 0.9
    # lr_temp = 0.02 * 1 / 16
    # lr_temp = 0.0001
    lr_temp = 0.0001
    weight_decay_ = 0.0001
    # device = 'cuda'
    print("Select Model Mode : 0 : train, 1: continue train, 2: bbox train 3: model test")
    select_mode = int(input())
    if select_mode == 0:
        model = Lidar_cluster_Rcnn(device, lr_temp, weight_decay_)
        mode = 'train'
    elif select_mode == 1:
        model = Lidar_cluster_Rcnn_continue(device, lr_temp, weight_decay_)
        mode = 'train'
    elif select_mode == 2:
        model = trainBBox(device, lr_temp, weight_decay_)
        mode = 'train'
    elif select_mode == 3:
        model = modelTest(device, lr_temp, weight_decay_)
        mode = 'test'


    if select_mode < 3:
        dataset = kitti_set(cfg.SRCPATH, 'training')
    elif select_mode >=3:
        dataset = kitti_set(cfg.SRCPATH, 'testing')

    dataset = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            shuffle=False, num_workers=0,
                                            collate_fn=collate_fn)      #

    start_epoch = 0
    end_epoch = 3

    for epoch in range(start_epoch, end_epoch):
        
        tm_1 = time.time()
        print((f"@@@[Epoch] : {epoch + 1}"))
        # train_one_epoch(model, optimizer, d_train, device, epoch)
        train_one_epoch(model, dataset, device, epoch, mode)
        # print('model save')

    final_loss = model.final_loss_list
    np.save('./save_final_loss', final_loss)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    main()
