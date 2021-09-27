from model.utils import collate_fn
from torch.nn import parameter
# from model.maskrcnn import MaskRCNN
# from load_data.coco import coco_set
from config import Config as cfg
# from model.detection.engine import train_one_epoch, evaluate
from Lidar_cluster_Rcnn import Lidar_cluster_Rcnn
import torch
# from load_data.kitti_loader import kitti_set
from load_data.kitti_loader_v100 import kitti_set
from train_one import train_one_epoch
from model.utils import collate_fn
from save_param import write_options

from train_continue import Lidar_cluster_Rcnn_continue
# from model_test import modelTest
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
now = time.localtime()


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

    dataset_train = kitti_set(cfg.SRCPATH, 'train')
    d_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, 
                                            shuffle=False, num_workers=0,
                                            collate_fn=collate_fn)
    momentum = 0.9
    lr_temp = 0.0001
    weight_decay_ = 0.0001

    start_epoch = 0
    end_epoch = 7
    folder = f"./result/{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}"
    os.makedirs(folder)
    file_name = f"./result/vgg16_model_7ep_{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}"
    write_options(file_name, 'Adam', lr_temp, weight_decay_, end_epoch, False)
    save_name = folder+"vgg16.pt"


    model = Lidar_cluster_Rcnn(device, lr_temp, weight_decay_, end_epoch)
    # model = Lidar_cluster_Rcnn_continue(device, lr_temp, weight_decay_, end_epoch)
    # model = modelTest(device, lr_temp, weight_decay_)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=lr_temp, weight_decay=weight_decay_)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(start_epoch, end_epoch):
        tm_1 = time.time()
        print((f"@@@[Epoch] : {epoch + 1}"))
        train_one_epoch(model, d_train, optimizer, device, epoch, save_name)
        lr_scheduler.step()


    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    main()
